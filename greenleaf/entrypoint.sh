#!/bin/sh
# Tham số thêm (vd --mem-fraction-static 0.85) truyền thẳng cho launch_server.
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION_STATIC" "$@"
