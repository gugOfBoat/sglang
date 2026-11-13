"""Cổng ① (không cần GPU): prompt SGLang đã vá == prompt lúc train, từng token và từng span.

  python check_render.py <sglang/srt đã vá> <thư mục model SGLang> <jsonl Jev> [--n 300]

Đường A = `encode_joint_schema` của SGLang (tokenizer transformers đọc từ thư mục model, như server).
Đường B = `Renderer.encode` của code train (tokenizers đọc tokenizer.json), thứ tự đáp án như lưu.
Request SGLang dựng theo hợp đồng API: choice gửi id A, B, C…; score gửi danh sách mô tả theo mức.
Đạt khi input_ids, span câu hỏi và span đáp án khớp 100%.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

GREENLEAF = Path(os.environ.get("GREENLEAF_SRC", "greenleaf-src"))  # code train GreenLeaf (repo content-safety: training/l2-greenguard/greenleaf)


def load_sglang(srt: Path):
    """Nạp joint_schema_head + joint_schema của SGLang, thay protocol bằng lớp tối giản."""
    for n in ["sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.pooler", "sglang.srt.entrypoints",
              "sglang.srt.entrypoints.systemone", "sglang.srt.entrypoints.systemone.protocol"]:
        sys.modules[n] = types.ModuleType(n)
    sys.modules["sglang.srt.layers.pooler"].EmbeddingPoolerOutput = object
    proto = sys.modules["sglang.srt.entrypoints.systemone.protocol"]

    @dataclass
    class Q:
        type: str
        instructions: object
        criteria: object = None

    proto.SystemOneNoulQuestion = type("SystemOneNoulQuestion", (Q,), {})
    proto.SystemOneChoiceQuestion = type("SystemOneChoiceQuestion", (Q,), {})
    proto.SystemOneScoreQuestion = type("SystemOneScoreQuestion", (Q,), {})
    proto.SystemOneQuestion = Q
    proto.SystemOneRequest = object
    mods = {}
    for name, rel in [("sglang.srt.layers.joint_schema_head", "layers/joint_schema_head.py"),
                      ("sglang.srt.entrypoints.systemone.joint_schema", "entrypoints/systemone/joint_schema.py")]:
        spec = importlib.util.spec_from_file_location(name, srt / rel)
        mods[name] = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    return proto, mods["sglang.srt.layers.joint_schema_head"], mods["sglang.srt.entrypoints.systemone.joint_schema"]


def to_request(rec: dict, proto, letters: str):
    qs = {}
    for q in rec["questions"]:
        if q["type"] == "noul":
            qs[q["key"]] = proto.SystemOneNoulQuestion("noul", q["instructions"])
        elif q["type"] == "choice":
            qs[q["key"]] = proto.SystemOneChoiceQuestion(
                "choice", q["instructions"], {letters[i]: o for i, o in enumerate(q["options"])})
        else:
            qs[q["key"]] = proto.SystemOneScoreQuestion(
                "score", q["instructions"], [q["anchors"][str(i)] for i in range(q["max"] + 1)])
    return types.SimpleNamespace(state=rec["state"], images=[], questions=qs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("srt", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=16384)
    a = ap.parse_args()

    proto, jsh, js = load_sglang(a.srt)
    sys.path.insert(0, str(GREENLEAF))
    from data_loader.jev import LETTERS  # noqa: E402
    from data_loader.renderer import Renderer  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(a.model)
    ren = Renderer(a.model, a.max_length, "sandwich")
    ok = bad = 0
    types_seen: dict[str, int] = {}
    lines = open(a.data).readlines()
    if True:
        for line in random.Random(0).sample(lines, min(a.n, len(lines))):
            rec = json.loads(line)
            for q in rec["questions"]:
                types_seen[q["type"]] = types_seen.get(q["type"], 0) + 1
            ids_b, qspan_b, ospan_b = ren.encode(rec, range(len(rec["questions"])), [None] * len(rec["questions"]))
            ids_a, layout = js.encode_joint_schema(tok, to_request(rec, proto, LETTERS), a.max_length, [], "sandwich")
            lq = jsh.parse_decision_layout(layout, len(ids_a))
            same = (ids_a == ids_b
                    and [tuple(q.question_span) for q in lq] == [tuple(s) for s in qspan_b]
                    and [[tuple(s) for s in q.option_spans] for q in lq] == [[tuple(s) for s in o] for o in ospan_b])
            if same:
                ok += 1
            else:
                bad += 1
                if bad <= 3:
                    k = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
                    print(f"LỆCH {rec['state_id']}: len {len(ids_a)} vs {len(ids_b)}, token đầu khác ở {k}: "
                          f"{tok.decode(ids_a[max(0, k - 5):k + 5])!r} vs {tok.decode(ids_b[max(0, k - 5):k + 5])!r}")
    print(json.dumps({"khớp": ok, "lệch": bad, "câu hỏi theo kiểu": types_seen}, ensure_ascii=False))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
