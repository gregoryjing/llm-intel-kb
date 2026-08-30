"""查看最近一次评估报告的逐条结果."""

import glob
import json

path = sorted(glob.glob("data/evals/eval_*.json"))[-1]
with open(path, encoding="utf-8") as f:
    r = json.load(f)
print("file:", path)
print("mode:", r["retriever_mode"], "| hit:", r["retrieval_hit_rate"], "| mrr:", r["mrr"])
for e in r["entries"]:
    srcs = ",".join(e["retrieved_sources"][:5])
    rank = e["first_hit_rank"] if e["first_hit_rank"] is not None else "-"
    print(f"{e['qa_id']}  hit={e['hit']!s:5s} rank={rank:>4}  {e['question'][:34]:36s} {srcs}")
