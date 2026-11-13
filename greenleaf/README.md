# greenleaf/: GreenGuard (GreenLeaf-Decision) trên SGLang

Nhánh `greenleaf/<bản upstream>` = tag upstream + các commit `[GreenLeaf]` + thư mục này. Head GreenLeaf khác head Clef
có sẵn trong SGLang ở 4 điểm (FP32, memory mix, SSMax, entmax), cộng thêm cách ghép prompt sandwich. Tất cả lấy từ
`GreenNode/GreenLeaf-Decision-9B` (`config.json` khối `greenleaf.head`, `modeling_greenleaf_decision.py`). Mọi thay
đổi chỉ bật khi `joint_head_config.json` của model có khối `greenleaf`.

| File | Để làm |
|---|---|
| `build.env` | bản upstream (tag, sha) + image gốc (digest) |
| `Dockerfile` | image gốc + áp commit + **kiểm tree hash = commit**; gắn label OCI (`revision` = sha) |
| `release.sh` | build từ một commit → tag `<upstream>-<sha12>` (+ tag `gl-*` nếu có) → smoke → push |
| `RELEASES.md` | sổ phát hành: mỗi image một dòng (commit, digest, checkpoint, kết quả kiểm) |
| `scripts/` | `to_sglang_format.py` (release → thư mục SGLang), cổng ① `check_render.py`, ② `check_head.py`, ③ `gate_gpu.py` (cần `GREENLEAF_SRC` = code train) |
| `smoke/` | 2 request mẫu + `smoke.sh` |

## Phát hành

```bash
MODEL_DIR=/data/.../GreenGuard-9B-sglang greenleaf/release.sh greenleaf/main   # build + smoke → image v0.5.23-<sha12>
greenleaf/release.sh greenleaf/main --push                # push (đã docker login)
# chép dòng in ra vào RELEASES.md, commit
```

Image đã push thì không ghi đè (release.sh từ chối nếu tag đã có trên registry).
Từ image ngược về commit: `docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' <image>`.

## Lên bản SGLang mới

```bash
git fetch upstream tag vX.Y.Z --no-tags
git checkout -b greenleaf/vX.Y.Z greenleaf/v0.5.23 && git rebase --onto vX.Y.Z v0.5.23
# sửa build.env (UPSTREAM_TAG, UPSTREAM_SHA, BASE_IMAGE kèm digest) → chạy cổng ①② (CPU) → release.sh → cổng ③ (GPU)
```
