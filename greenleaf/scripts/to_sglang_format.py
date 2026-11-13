"""Chuyển thư mục release GreenLeaf/GreenGuard (đầu ra export_release.py) sang định dạng SGLang đọc được.

  python to_sglang_format.py <thư mục release> <thư mục ra>

Release GreenLeaf                         →  thư mục cho SGLang
  config.json (qwen3_5_text, phẳng)       →  config.json: Qwen3_5ForConditionalGeneration, text_config,
                                             language_model_only=true (không dựng vision tower)
  layers.* / embed_tokens / norm          →  model.language_model.*   (tên SGLang map được)
  lexical.weight                          →  lm_head.weight           (SGLang lấy lm_head làm bảng lexical)
  head.* (fp32, lẫn trong shard)          →  joint_head.safetensors   (bỏ tiền tố head.)
  config.greenleaf.head + joint_head_dims →  joint_head_config.json   (kèm khoá "greenleaf": SGLang chưa vá sẽ
                                             báo lỗi khi tạo head thay vì âm thầm cho điểm sai)
  tokenizer + chat_template               →  chép nguyên
  + SHA256SUMS, conversion_info.json

Chỉ đổi tên và tách file, không đổi giá trị tensor nào. Chạy trên CPU.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

COPY = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "generation_config.json", "LICENSE"]
TEXT_DROP = {"architectures", "auto_map", "greenleaf"}


def backbone_name(k: str) -> str | None:
    if k == "lexical.weight":
        return "lm_head.weight"
    if k.startswith("head."):
        return None
    if k.startswith(("layers.", "embed_tokens.", "norm.")):
        return "model.language_model." + k
    raise ValueError(f"tên tensor lạ: {k}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main(src: Path, dst: Path) -> None:
    if dst.exists():
        raise SystemExit(f"{dst} đã có; mỗi version một thư mục, không ghi đè")
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("model_type") != "qwen3_5_text" or "greenleaf" not in cfg:
        raise SystemExit("không phải release GreenLeaf (model_type qwen3_5_text + khoá greenleaf)")
    gl = cfg["greenleaf"]
    dst.mkdir(parents=True)

    text = {k: v for k, v in cfg.items() if k not in TEXT_DROP}
    out_cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
               "language_model_only": True, "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
               "text_config": text}
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2, ensure_ascii=False) + "\n")

    head_cfg = {k: v for k, v in gl["head"].items() if k not in ("kind", "clef_dir")}
    (dst / "joint_head_config.json").write_text(
        json.dumps({**gl["joint_head_dims"], "greenleaf": head_cfg}, indent=2) + "\n")

    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    new_map, head, seen = {}, {}, 0
    for shard in sorted(set(wm.values())):
        tensors = {}
        with safe_open(src / shard, "pt") as f:
            for k in f.keys():
                seen += 1
                name = backbone_name(k)
                if name is None:
                    head[k[len("head."):]] = f.get_tensor(k)
                else:
                    tensors[name] = f.get_tensor(k)
        if tensors:
            save_file(tensors, dst / shard, metadata={"format": "pt"})
            new_map.update(dict.fromkeys(tensors, shard))
        print(f"{shard}: {len(tensors)} tensor backbone", flush=True)
    if seen != len(wm):
        raise SystemExit(f"đọc {seen} tensor, index ghi {len(wm)}")
    if "lm_head.weight" not in new_map:
        raise SystemExit("thiếu lexical.weight")
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict(sorted(new_map.items()))}, indent=2) + "\n")
    save_file(head, dst / "joint_head.safetensors", metadata={"format": "pt"})

    for name in COPY:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    info = {"source": str(src), "backbone_tensors": len(new_map), "head_tensors": len(head)}
    if (src / "export_info.json").exists():
        info["export_info"] = json.loads((src / "export_info.json").read_text())
    (dst / "conversion_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    sums = [f"{sha256(p)}  {p.name}" for p in sorted(dst.iterdir()) if p.is_file()]
    (dst / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(info["backbone_tensors"]), "backbone +", len(head), "head →", dst)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
