#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONTAINER_NAME="${CONTAINER_NAME:-qdrant}"
IMAGE_NAME="${IMAGE_NAME:-qdrant/qdrant}"
PORT="${PORT:-6333}"
STORAGE_DIR="${STORAGE_DIR:-$ROOT_DIR/qdrant_storage}"

mkdir -p "$STORAGE_DIR"

echo "Starting Qdrant container '$CONTAINER_NAME' on port $PORT..."

podman run -d \
  --name "$CONTAINER_NAME" \
  -p "${PORT}:6333" \
  -v "${STORAGE_DIR}:/qdrant/storage" \
  "$IMAGE_NAME"

echo "Qdrant started. You can check it with:"
echo "  curl http://localhost:${PORT}/collections"

