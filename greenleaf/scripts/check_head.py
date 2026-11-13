"""Cổng ② phần CPU: head SGLang đã vá cho cùng logit với head lúc train, trên cùng một hidden giả.

  python check_head.py <sglang/srt đã vá> <release GreenLeaf> <thư mục SGLang> <jsonl Jev> [--n 64] [--batch 8]

Hai đường nhận đúng một input: hidden cuối và hidden các layer trộn (random, bf16, cố định theo seed),
input_ids thật của prompt.
- Đường train: `model.head.JointSchemaHead` (code train), batch đệm phải như `Collate`, chạy dưới autocast bf16.
- Đường SGLang: `JointSchemaHead.forward_batch` đã vá, batch nối liền như server, đọc `joint_head_config.json`
  và `joint_head.safetensors` của thư mục SGLang (tức là kiểm luôn bước chuyển định dạng).

Không cần GPU. Trên CPU, attention của SGLang đi đường tính tay (`_dense_attention`), không qua flash; đường
flash chỉ kiểm được trên GPU (cổng ② trên máy 5090).

Ngoài bản vá đầy đủ (`full`), script chạy thêm các biến thể bỏ bớt một chỗ vá, để thấy mỗi chỗ ảnh hưởng tới
đâu. Đạt khi `full` lật 0 câu và `solo` (mỗi request chạy riêng) khớp `full` (chạy gộp).
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_render import GREENLEAF, load_sglang, to_request  # noqa: E402


class _Records:
    def __init__(self, recs):
        self.recs = recs

    def record(self, s):
        return self.recs[s]


def sglang_head(jsh, sg_dir: Path, num_layers: int):
    """Dựng head như `Qwen3_5ForConditionalGeneration` đã vá, nạp strict, FP32."""
    cfg = json.loads((sg_dir / "joint_head_config.json").read_text())
    dims = {k: v for k, v in cfg.items() if k != "greenleaf"}
    gl = cfg.get("greenleaf") or {}
    n_mix = 0
    if gl.get("memory_layers_from") is not None:
        import math
        n_mix = num_layers - 1 - (math.ceil(gl["memory_layers_from"] * num_layers) - 1)
    head = jsh.JointSchemaHead(**dims, n_mix=n_mix, entmax=gl.get("entmax", False),
                               entmax_alpha_init=gl.get("entmax_alpha_init", 1.01), ssmax=gl.get("ssmax", False))
    head.float()
    head.load_state_dict(load_file(sg_dir / "joint_head.safetensors"), strict=True)
    return head.eval()


def variants(head):
    """Bản vá đầy đủ + các biến thể bỏ một chỗ vá."""
    def mem_attn(h):
        return [l.attention for l in h.evidence_layers] + [l.multihead_attn for l in h.layers]

    out = {"full": head}
    h = copy.deepcopy(head); h.entmax_softmax = True; out["entmax→softmax"] = h
    h = copy.deepcopy(head); h.memory_mix = None; out["bỏ memory mix"] = h
    h = copy.deepcopy(head)
    for m in mem_attn(h):
        m.ssmax_beta = None
    out["bỏ ssmax"] = h
    h = copy.deepcopy(head); h.greenleaf = False; h.bfloat16(); out["head bf16 (không vá fp32)"] = h
    h = copy.deepcopy(head); h.greenleaf = False; h.memory_mix = None; h.entmax_softmax = True
    for m in mem_attn(h):
        m.ssmax_beta = None
    h.bfloat16()
    out["chưa vá gì (SGLang gốc)"] = h
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("srt", type=Path)
    ap.add_argument("release", type=Path)
    ap.add_argument("sglang_dir", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=16384)
    a = ap.parse_args()

    proto, jsh, js = load_sglang(a.srt)
    sys.path.insert(0, str(GREENLEAF))
    from data_loader.jev import LETTERS  # noqa: E402
    from data_loader.renderer import Collate, Renderer  # noqa: E402
    from model.head import build_head  # noqa: E402
    from model.model import read_head_config, read_shard_weights  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    torch.set_num_threads(min(16, torch.get_num_threads()))
    head_config, dims = read_head_config(a.release)
    state, lexical = read_shard_weights(a.release)
    num_layers = json.loads((a.sglang_dir / "config.json").read_text())["text_config"]["num_hidden_layers"]
    train_head = build_head(head_config, dims, num_layers, state).eval()
    n_mix = 0 if train_head.memory_mix is None else len(train_head.memory_mix)
    heads = variants(sglang_head(jsh, a.sglang_dir, num_layers))

    tok = AutoTokenizer.from_pretrained(a.sglang_dir)
    ren = Renderer(a.release, a.max_length, "sandwich")
    lines = open(a.data).readlines()
    recs = [json.loads(x) for x in random.Random(a.seed).sample(lines, min(a.n, len(lines)))]
    ds = _Records(recs)
    gen = torch.Generator().manual_seed(a.seed)
    hidden_size = dims["hidden_size"]

    stats = {k: {"câu": 0, "lật": 0, "max_dlogit": 0.0, "max_dp": 0.0} for k in [*heads, "solo vs gộp"]}
    types_seen: dict[str, int] = {}

    def compare(name, ref, got):
        s = stats[name]
        for r, g in zip(ref, got):
            s["câu"] += 1
            s["lật"] += int(r.argmax() != g.argmax())
            s["max_dlogit"] = max(s["max_dlogit"], (r - g).abs().max().item())
            s["max_dp"] = max(s["max_dp"], (r.softmax(-1) - g.softmax(-1)).abs().max().item())

    for g0 in range(0, len(recs), a.batch):
        idx = list(range(g0, min(g0 + a.batch, len(recs))))
        for i in idx:
            for q in recs[i]["questions"]:
                types_seen[q["type"]] = types_seen.get(q["type"], 0) + 1
        # Đường SGLang: prompt + layout từ encode_joint_schema
        ids_list, layouts = [], []
        for i in idx:
            ids, layout = js.encode_joint_schema(tok, to_request(recs[i], proto, LETTERS), a.max_length, [], "sandwich")
            ids_list.append(ids)
            layouts.append(jsh.parse_decision_layout(layout, len(ids)))
        lengths = [len(x) for x in ids_list]
        hid = [torch.randn(n, hidden_size, generator=gen).to(torch.bfloat16) for n in lengths]
        mids = [[torch.randn(n, hidden_size, generator=gen).to(torch.bfloat16) for n in lengths] for _ in range(n_mix)]

        # Đường train: Collate (thứ tự đáp án như lưu) + đệm phải
        units = [(i, tuple(range(len(recs[i]["questions"])))) for i in idx]
        b = Collate(ds, ren, units, epoch_seed=None, eval_seed=None)(list(range(len(units))))
        width = b["input_ids"].shape[1]
        for r, ids in enumerate(ids_list):
            assert b["input_ids"][r, :len(ids)].tolist() == ids, "prompt lệch (xem cổng ①)"
        hp = torch.zeros(len(idx), width, hidden_size, dtype=torch.bfloat16)
        mp = [torch.zeros(len(idx) * width, hidden_size, dtype=torch.bfloat16) for _ in range(n_mix)]
        for r, n in enumerate(lengths):
            hp[r, :n] = hid[r]
            for j in range(n_mix):
                mp[j][r * width:r * width + n] = mids[j][r]
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            flat = train_head(hp, mp, b["input_ids"], lexical, b).float()
        ref = [flat[b["o_q"] == q] for q in range(len(b["q_row"]))]

        flat_ids = torch.tensor([t for x in ids_list for t in x])
        cat_mids = [torch.cat(m) for m in mids]
        for name, h in heads.items():
            with torch.no_grad():
                out = h.forward_batch(torch.cat(hid), flat_ids, lengths, layouts, lexical,
                                      cat_mids if h.memory_mix is not None else None)
            got = [q for v, lay in zip(out, layouts) for q in torch.split(v, [len(x.option_spans) for x in lay])]
            compare(name, ref, got)
            if name == "full":
                full = got
        # Từng request chạy riêng (batch 1) so với chạy gộp
        solo = []
        for r in range(len(idx)):
            with torch.no_grad():
                v = heads["full"].forward_batch(hid[r], torch.tensor(ids_list[r]), [lengths[r]], [layouts[r]],
                                                lexical, [m[r] for m in mids])[0]
            solo += list(torch.split(v, [len(x.option_spans) for x in layouts[r]]))
        compare("solo vs gộp", full, solo)
        print(f"{idx[-1] + 1}/{len(recs)}", file=sys.stderr, flush=True)

    ok = stats["full"]["lật"] == 0 and stats["solo vs gộp"]["lật"] == 0
    print(json.dumps({"đạt": ok, "mẫu": len(recs), "câu hỏi theo kiểu": types_seen, "token": "random seed "
                      f"{a.seed}", "kết quả": stats}, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
