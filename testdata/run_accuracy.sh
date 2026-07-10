#!/usr/bin/env bash
set -u
set -o pipefail

DATASET="${1:-all}"
NUM_ROWS="${2:-}"

case "$DATASET" in
  all|hotpotqa|gov_report|retrieval_multi_point|aggregation_keyword_aggregation) ;;
  *)
    echo "usage: $0 [all|hotpotqa|gov_report|retrieval_multi_point|aggregation_keyword_aggregation] [num_rows]"
    exit 1
    ;;
esac

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export OPENAI_API_KEY=EMPTY
export TOKENIZERS_PARALLELISM=false
export SELECTED_DATASET="$DATASET"

mkdir -p ./accuracy_debug/data
rm -f ./accuracy_debug/data/*.jsonl

selected() {
  [ "$DATASET" = "all" ] || [ "$DATASET" = "$1" ]
}

prepare_dataset() {
  local name="$1"
  local src="./${name}.jsonl"
  local dst="./accuracy_debug/data/${name}.jsonl"

  if ! selected "$name"; then
    return 0
  fi

  if [ -n "$NUM_ROWS" ]; then
    head -n "$NUM_ROWS" "$src" > "$dst"
  else
    cp "$src" "$dst"
  fi

  echo "dataset=${name} rows=$(wc -l < "$dst")"
}

prepare_dataset hotpotqa
prepare_dataset gov_report
prepare_dataset retrieval_multi_point
prepare_dataset aggregation_keyword_aggregation

cat > ./accuracy_debug/run.py <<'PY'
from opencompass.cli.main import main

if __name__ == "__main__":
    main()
PY

cat > ./accuracy_debug/bench.py <<'PY'
import os

from opencompass.datasets import CustomDataset
from opencompass.models import OpenAISDK
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen_6b3efc import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen_54c5b0 import LongBench_gov_report_datasets

DATA_DIR = "./data"
SELECTED = os.environ.get("SELECTED_DATASET", "all")
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.abspath(os.path.join(os.getcwd(), "..", "..", "Qwen3.5-27B")),
)


def enabled(name):
    return SELECTED == "all" or SELECTED == name


def build_longbench_local(base_cfg, file_name, abbr, max_out_len):
    d = dict(base_cfg)
    d["abbr"] = abbr
    d["type"] = CustomDataset
    d["path"] = os.path.join(DATA_DIR, file_name)
    d["local_mode"] = True
    d.pop("name", None)
    d["infer_cfg"] = dict(d.get("infer_cfg", {}))
    d["infer_cfg"]["inferencer"] = dict(d["infer_cfg"].get("inferencer", {}))
    d["infer_cfg"]["inferencer"]["max_out_len"] = max_out_len
    return d


def build_ruler_dataset(file_name, abbr, max_out_len):
    return dict(
        abbr=abbr,
        type=CustomDataset,
        path=os.path.join(DATA_DIR, file_name),
        local_mode=True,
        reader_cfg=dict(
            input_columns=["prompt"],
            output_column="target_text",
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[dict(role="HUMAN", prompt="{prompt}")]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=max_out_len),
        ),
        eval_cfg=dict(
            evaluator=dict(type=AccEvaluator),
            pred_role="BOT",
        ),
    )


datasets = []

if enabled("hotpotqa"):
    datasets.append(build_longbench_local(
        LongBench_hotpotqa_datasets[0], "hotpotqa.jsonl", "hotpotqa", 1024))

if enabled("gov_report"):
    datasets.append(build_longbench_local(
        LongBench_gov_report_datasets[0], "gov_report.jsonl", "gov_report", 1024))

if enabled("retrieval_multi_point"):
    datasets.append(build_ruler_dataset(
        "retrieval_multi_point.jsonl", "retrieval_multi_point", 96))

if enabled("aggregation_keyword_aggregation"):
    datasets.append(build_ruler_dataset(
        "aggregation_keyword_aggregation.jsonl", "aggregation_keyword_aggregation", 128))

work_dir = "./output/local_accuracy_qwen35"

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

models = [
    dict(
        abbr="Qwen3.5-27B-vLLM",
        type=OpenAISDK,
        path="Qwen3.5-27B",
        openai_api_base="http://127.0.0.1:8001/v1",
        tokenizer_path=MODEL_DIR,
        key="EMPTY",
        meta_template=api_meta_template,
        temperature=0,
        query_per_second=1,
        max_out_len=1024,
        max_seq_len=32768,
        pred_postprocessor=dict(
            type="opencompass.utils.text_postprocessors.extract_non_reasoning_content"
        ),
        batch_size=2,
        verbose=True,
    )
]

del enabled
del build_longbench_local
del build_ruler_dataset
del LongBench_hotpotqa_datasets
del LongBench_gov_report_datasets
del MODEL_DIR
del DATA_DIR
del SELECTED
del os
del CustomDataset
del OpenAISDK
del AccEvaluator
del GenInferencer
del PromptTemplate
del ZeroRetriever
del read_base
PY

cd ./accuracy_debug
python run.py bench.py --debug > opencompass_run.log 2>&1

LATEST_RUN="$(ls -td ./output/local_accuracy_qwen35/* 2>/dev/null | head -n1)"
export LATEST_RUN

if [ -z "$LATEST_RUN" ]; then
  echo "no output dir generated"
  exit 1
fi

python3 - <<'PY2'
from pathlib import Path
from collections import Counter
import csv
import json
import os
import re

ROOT = Path(os.environ["LATEST_RUN"])
DATA = Path("./data")
MODEL = "Qwen3.5-27B-vLLM"
SELECTED = os.environ.get("SELECTED_DATASET", "all")

def enabled(name):
    return SELECTED == "all" or SELECTED == name

def norm(x):
    x = "" if x is None else str(x)
    x = x.strip()
    x = re.sub(r"^```(?:json)?\s*", "", x)
    x = re.sub(r"\s*```$", "", x)
    return x.strip().strip('"').strip("'")

def parse_pred_list(text):
    text = norm(text)
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [norm(i) for i in obj if norm(i)]
        if isinstance(obj, str):
            return [norm(obj)] if norm(obj) else []
    except Exception:
        pass
    if "\n" in text:
        return [norm(i) for i in text.splitlines() if norm(i)]
    if "," in text:
        return [norm(i) for i in text.split(",") if norm(i)]
    return [text]

def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]

def normalize_predictions(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("predictions"), list):
            return raw["predictions"]
        keys = []
        for k in raw:
            try:
                keys.append((int(k), k))
            except Exception:
                pass
        if keys:
            return [raw[k] for _, k in sorted(keys)]
    return []

def match(gold_list, pred_text):
    gold = [norm(i) for i in gold_list if norm(i)]
    pred = parse_pred_list(pred_text)
    return Counter(pred) == Counter(gold)

def score_ruler(name):
    pred_path = ROOT / "predictions" / MODEL / f"{name}.json"
    gold_path = DATA / f"{name}.jsonl"

    if not pred_path.exists():
        return None

    gold_rows = load_jsonl(gold_path)
    raw = json.loads(pred_path.read_text(encoding="utf-8", errors="ignore"))
    pred_rows = normalize_predictions(raw)

    total = min(len(gold_rows), len(pred_rows))
    ok = 0

    for i in range(total):
        gold = gold_rows[i]
        pred = pred_rows[i]
        if isinstance(pred, dict):
            pred_text = pred.get("prediction") or pred.get("pred") or pred.get("text") or pred.get("output") or ""
        else:
            pred_text = pred

        gold_list = gold.get("target_list") or [gold.get("target_text", "")]
        ok += int(match(gold_list, pred_text))

    score = ok / total * 100 if total else 0.0
    return f"{score:.2f}"

def read_summary_rows():
    csv_files = sorted((ROOT / "summary").glob("summary_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        return []

    with open(csv_files[-1], encoding="utf-8", errors="ignore") as f:
        all_rows = list(csv.reader(f))

    if not all_rows:
        return []

    header = all_rows[0]
    model_col = header.index(MODEL) if MODEL in header else len(header) - 1

    rows = []
    for row in all_rows[1:]:
        if not row:
            continue

        name = row[0].strip()
        if not enabled(name):
            continue

        version = row[1].strip() if len(row) > 1 else "-"
        metric = row[2].strip() if len(row) > 2 else "-"
        mode = row[3].strip() if len(row) > 3 else "-"
        score = row[model_col].strip() if len(row) > model_col else "-"

        if name in {"retrieval_multi_point", "aggregation_keyword_aggregation"}:
            recalculated = score_ruler(name)
            if recalculated is not None:
                score = recalculated

        rows.append((name, version, metric, mode, score))

    return rows

rows = read_summary_rows()

print()
print("===== Final Accuracy Results =====")
print("| dataset | version | metric | mode | Qwen3.5-27B-vLLM |")
print("|----- | ----- | ----- | ----- | -----|")
for name, version, metric, mode, score in rows:
    print(f"| {name} | {version} | {metric} | {mode} | {score} |")

print()
print(f"output_dir: {ROOT}")
PY2