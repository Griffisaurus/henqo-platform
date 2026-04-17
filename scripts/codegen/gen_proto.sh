#!/usr/bin/env bash
# Generate Python protobuf bindings from all .proto files in packages/schema/proto/
# Usage: scripts/codegen/gen_proto.sh
set -euo pipefail

PROTO_DIR="packages/schema/proto"
OUT_DIR="packages/schema/src/generated"

echo "Generating Python protobuf bindings..."
echo "  Source: $PROTO_DIR"
echo "  Output: $OUT_DIR"

mkdir -p "$OUT_DIR"

python -m grpc_tools.protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR"/*.proto

# Fix relative imports in generated grpc files (grpc_tools generates absolute imports)
for f in "$OUT_DIR"/*_pb2_grpc.py; do
  if [ -f "$f" ]; then
    # Replace "import <module>_pb2" with "from . import <module>_pb2"
    sed -i 's/^import \([a-zA-Z_]*\)_pb2/from . import \1_pb2/' "$f"
  fi
done

# Create __init__.py if not present
touch "$OUT_DIR/__init__.py"

echo "Done. Generated files:"
ls "$OUT_DIR"/*.py 2>/dev/null | sed 's|^|  |'
