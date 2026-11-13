"""Cổng ③ (GPU): server SGLang đã vá cho cùng xác suất với code eval lúc train, trên model thật.

  python gate_gpu.py sample  <jsonl Jev> <out.jsonl> [--n 300 --longest 30 --seed 0]
  python gate_gpu.py ref     <records.jsonl> <release GreenLeaf> <out.jsonl>       # venv code train, cần GPU
  python gate_gpu.py serve   <records.jsonl> <url> <out.jsonl> [--concurrency 1]   # chỉ cần python3
  python gate_gpu.py compare <ref.jsonl> <sglang.jsonl> [<sglang2.jsonl> ...]

- sample: n bản ghi ngẫu nhiên + `longest` bản ghi có state dài nhất (để thử cả prompt dài).
- ref: đường B của verify_release (code eval, chỉ bản phát hành), mỗi bản ghi chạy riêng.
- serve: gửi từng bản ghi tới `/v1/systemone` (choice gửi id A, B, C…, score gửi danh sách mức).
  `--concurrency` > 1 để server gộp nhiều request trong một prefill.
- compare: chênh xác suất lớn nhất / trung bình, số câu đổi đáp án (argmax) và số câu noul đổi phía ngưỡng 0,5;
  tách theo độ dài prompt. Đạt khi 0 câu đổi đáp án.
Mọi xác suất ghi theo thứ tự lớp của code train (noul: [false, true]; choice: thứ tự lưu; score: mức 0…K).
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
GREENLEAF = Path(os.environ.get("GREENLEAF_SRC", "greenleaf-src"))  # code train GreenLeaf (repo content-safety: training/l2-greenguard/greenleaf)


def sample(a):
    lines = open(a.data).readlines()
    rng = random.Random(a.seed)
    picked = set(rng.sample(range(len(lines)), min(a.n, len(lines))))
    by_len = sorted(range(len(lines)), key=lambda i: -len(lines[i]))
    picked |= set(by_len[:a.longest])
    with open(a.out, "w") as f:
        for i in sorted(picked):
            f.write(lines[i])
    print(f"{len(picked)} bản ghi → {a.out}")


def ref(a):
    import torch
    sys.path.insert(0, str(GREENLEAF))
    import model.model as module_arch
    from data_loader.jev import to_device
    from data_loader.renderer import Collate, Renderer

    class One:
        def __init__(self, r):
            self.r = r

        def record(self, s):
            return self.r

    head_config, _ = module_arch.read_head_config(a.release)
    renderer = Renderer(a.release, 16384, layout=head_config["layout"])
    dev = torch.device("cuda")
    model = module_arch.GreenLeafDecisionModel.build(a.release, dev).eval()
    with open(a.out, "w") as f:
        for line in open(a.records):
            r = json.loads(line)
            b = Collate(One(r), renderer, [(0, tuple(range(len(r["questions"]))))], epoch_seed=None, eval_seed=None)([0])
            with torch.no_grad():
                cl = model(to_device(b, dev)).float().cpu()
            probs = [torch.softmax(cl[i, :int(b["k"][i])], -1).tolist() for i in range(len(r["questions"]))]
            f.write(json.dumps({"state_id": r["state_id"], "tokens": int(b["n_tokens"]), "probs": probs}) + "\n")
    print(f"xong → {a.out}")


def to_request(r: dict, model: str) -> dict:
    qs = {}
    for q in r["questions"]:
        if q["type"] == "noul":
            qs[q["key"]] = {"type": "noul", "instructions": q["instructions"]}
        elif q["type"] == "choice":
            qs[q["key"]] = {"type": "choice", "instructions": q["instructions"],
                            "criteria": {LETTERS[i]: o for i, o in enumerate(q["options"])}}
        else:
            qs[q["key"]] = {"type": "score", "instructions": q["instructions"],
                            "criteria": [q["anchors"][str(i)] for i in range(q["max"] + 1)]}
    return {"model": model, "state": r["state"], "questions": qs}


def from_answer(q: dict, ans: dict) -> list:
    if q["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    if q["type"] == "choice":
        return [ans["probabilities"][LETTERS[i]] for i in range(len(q["options"]))]
    return [ans["probabilities"][str(i)] for i in range(q["max"] + 1)]


def serve(a):
    recs = [json.loads(x) for x in open(a.records)]

    def one(r):
        body = json.dumps(to_request(r, a.model)).encode()
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", body, {"Content-Type": "application/json"})
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = json.loads(resp.read())
        ms = (time.perf_counter() - t) * 1000
        probs = [from_answer(q, out["answers"][q["key"]]) for q in r["questions"]]
        return {"state_id": r["state_id"], "tokens": out["usage"]["input_tokens"], "ms": round(ms, 1), "probs": probs}

    with ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(one, recs))
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(rows)} request, concurrency {a.concurrency} → {a.out}")


def compare(a):
    ref_rows = {r["state_id"]: r for r in map(json.loads, open(a.ref))}
    buckets = [(0, 1024), (1024, 4096), (4096, 16385)]
    result = {}
    for path in a.others:
        stat = {"câu": 0, "đổi đáp án": 0, "noul đổi phía 0,5": 0, "max_dp": 0.0, "sum_dp": 0.0,
                "theo độ dài": {f"{lo}-{hi - 1}": {"câu": 0, "đổi": 0, "max_dp": 0.0} for lo, hi in buckets},
                "lật": []}
        for row in map(json.loads, open(path)):
            rr = ref_rows[row["state_id"]]
            if rr["tokens"] != row["tokens"]:
                stat.setdefault("token lệch", []).append([row["state_id"], rr["tokens"], row["tokens"]])
            bk = next(f"{lo}-{hi - 1}" for lo, hi in buckets if lo <= rr["tokens"] < hi)
            for qi, (p, g) in enumerate(zip(rr["probs"], row["probs"])):
                d = max(abs(x - y) for x, y in zip(p, g))
                flip = max(range(len(p)), key=p.__getitem__) != max(range(len(g)), key=g.__getitem__)
                stat["câu"] += 1
                stat["đổi đáp án"] += flip
                if len(p) == 2:
                    stat["noul đổi phía 0,5"] += (p[1] >= 0.5) != (g[1] >= 0.5)
                stat["max_dp"] = max(stat["max_dp"], d)
                stat["sum_dp"] += d
                b = stat["theo độ dài"][bk]
                b["câu"] += 1
                b["đổi"] += flip
                b["max_dp"] = max(b["max_dp"], d)
                if flip:
                    stat["lật"].append({"state_id": row["state_id"], "q": qi, "tokens": rr["tokens"],
                                        "ref": [round(x, 4) for x in p], "sglang": [round(x, 4) for x in g]})
        stat["mean_dp"] = stat.pop("sum_dp") / max(stat["câu"], 1)
        result[path] = stat
    ok = all(s["đổi đáp án"] == 0 for s in result.values())
    print(json.dumps({"đạt": ok, **result}, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("data"); p.add_argument("out")
    p.add_argument("--n", type=int, default=300); p.add_argument("--longest", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("ref"); p.add_argument("records"); p.add_argument("release"); p.add_argument("out")
    p = sub.add_parser("serve"); p.add_argument("records"); p.add_argument("url"); p.add_argument("out")
    p.add_argument("--concurrency", type=int, default=1); p.add_argument("--model", default="greenguard-9b")
    p = sub.add_parser("compare"); p.add_argument("ref"); p.add_argument("others", nargs="+")
    a = ap.parse_args()
    {"sample": sample, "ref": ref, "serve": serve, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
