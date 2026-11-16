# Sổ phát hành image

Mỗi image đã push một dòng. Không sửa dòng cũ; image sai thì thêm dòng mới và ghi lý do.

| Ngày (UTC) | Version | Commit | Upstream | Digest | Checkpoint đã kiểm | Cổng ③ (so code train) |
|---|---|---|---|---|---|---|
| 2026-11-16 | `v0.5.23-e819e08a59f4` | `e819e08a59f4` | v0.5.23 | `sha256:2611dc1f473ccbae3e0425ee8588123bc8e99260c4a27fbcbe9a806d53b42126` | GreenGuard-9B s13600 (1.0) | 0/2.206 lật ở CCU1 và CCU64 (2× RTX PRO 6000 Max-Q) |
| 2026-11-16 | `v0.5.23-84fbd2f60ca3` | `84fbd2f60ca3` | v0.5.23 | `sha256:d2d7f297239745ce046525ef115606922db8a8b434321ed976fafb9ddf83569b` | GreenGuard-9B s13600 (1.0) | 0/2.206 lật; CCU1 trùng bit với e819 (RTX PRO 6000 Max-Q) |
