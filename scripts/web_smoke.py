"""Web 冒烟测试：验证新端点与混合检索状态."""

import httpx2 as httpx

BASE = "http://127.0.0.1:8766"

with httpx.Client(timeout=30) as c:
    st = c.get(f"{BASE}/api/status").json()
    print("index embedding:", st["index"].get("embedding_model"), st["index"].get("embedding_dim"))
    print("index_ready:", st["index_ready"], "| chunks:", st["index"]["chunks"])

    hist = c.get(f"{BASE}/api/eval/history").json()
    print(f"\neval history: {len(hist['runs'])} runs")
    for run in hist["runs"]:
        mode = run["retriever_mode"][:30]
        print(
            f"  {run['created_at'][:16]}  hit={run['retrieval_hit_rate']:.2f}"
            f"  mrr={run['mrr']:.3f}  {mode}"
        )

    r = c.post(f"{BASE}/api/search", json={"query": "大语言模型推理优化", "k": 3})
    data = r.json()
    print(f"\nsearch mode: {data['mode']}")
    for hit in data["hits"]:
        tags = []
        if hit["lexical_rank"]:
            tags.append(f"词法#{hit['lexical_rank']}")
        if hit["vector_rank"]:
            tags.append(f"向量#{hit['vector_rank']}")
        print(f"  [{hit['n']}] {hit['title'][:44]}  {'/'.join(tags)}  rrf={hit['rrf_score']}")
