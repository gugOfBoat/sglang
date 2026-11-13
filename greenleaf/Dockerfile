# GreenGuard / GreenLeaf-Decision trên SGLang. Không build SGLang lại: lấy image chính thức của đúng bản upstream,
# áp các commit của nhánh greenleaf lên repo git có sẵn trong image, rồi kiểm tree hash = commit được build.
# Build bằng greenleaf/release.sh (tự tạo build context và truyền ARG), không build tay.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG UPSTREAM_TAG
ARG UPSTREAM_SHA
ARG GIT_SHA
ARG GIT_TREE
ARG VERSION
ARG CREATED
ARG SOURCE_URL
COPY patches/ /tmp/greenleaf-patches/
RUN set -e; cd /sgl-workspace/sglang \
 && test "$(git rev-parse HEAD)" = "${UPSTREAM_SHA}" || { echo "image gốc không ở ${UPSTREAM_SHA}"; exit 1; } \
 && test -z "$(git status --porcelain)" || { echo "repo trong image gốc có thay đổi lạ"; exit 1; } \
 && for p in /tmp/greenleaf-patches/*.patch; do git apply --index "$p"; done \
 && test "$(git write-tree)" = "${GIT_TREE}" || { echo "tree sau khi áp != commit ${GIT_SHA}"; exit 1; } \
 && rm -rf /tmp/greenleaf-patches \
 && python -c "import sglang.srt.layers.joint_schema_head, sglang.srt.entrypoints.systemone.joint_schema"
LABEL org.opencontainers.image.title="greenguard-sglang" \
      org.opencontainers.image.description="SGLang ${UPSTREAM_TAG} + GreenLeaf decision head" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      vn.greennode.greenleaf.upstream="${UPSTREAM_TAG}@${UPSTREAM_SHA}" \
      vn.greennode.greenleaf.tree="${GIT_TREE}"
# MEM_FRACTION_STATIC 0.65: model decision chỉ prefill, head cần VRAM tạm (hidden 12 layer); 0.78 mặc định OOM ở
# 16 request song song trên RTX 5090 32 GB.
ENV MODEL_PATH=/model SERVED_MODEL_NAME=greenguard-9b PORT=30000 CONTEXT_LENGTH=16640 MEM_FRACTION_STATIC=0.65 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GREENLEAF_GIT_SHA=${GIT_SHA} GREENLEAF_VERSION=${VERSION}
EXPOSE 30000
ENTRYPOINT ["/sgl-workspace/sglang/greenleaf/entrypoint.sh"]
