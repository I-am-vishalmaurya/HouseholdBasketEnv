#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Push the household_basket_env Docker image to a HuggingFace Space.
# Mirrors the deploy flow used in module1/food_label_auditor (plan §11).
#
# Usage:
#   HF_USERNAME=<your-hf-handle> ./scripts/push_to_hf.sh [tag]
#
# Optional env vars:
#   HF_TOKEN        - if not set, falls back to `huggingface-cli login`
#   HF_SPACE_NAME   - default: household-basket-env
#   IMAGE_TAG       - default: latest
# -----------------------------------------------------------------------------
set -euo pipefail

if ! command -v openenv >/dev/null 2>&1; then
  echo "[error] 'openenv' CLI not found. Install via: pip install openenv-core[cli]" >&2
  exit 1
fi

HF_SPACE_NAME=${HF_SPACE_NAME:-household-basket-env}
IMAGE_TAG=${1:-${IMAGE_TAG:-latest}}

if [[ -z "${HF_USERNAME:-}" ]]; then
  echo "[error] HF_USERNAME env var is required (your HuggingFace handle)" >&2
  exit 1
fi

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[info] env_dir = ${ENV_DIR}"
echo "[info] target  = hf://${HF_USERNAME}/${HF_SPACE_NAME}:${IMAGE_TAG}"

cd "${ENV_DIR}"

echo "[info] building image ..."
docker build -t "household_basket_env:${IMAGE_TAG}" .

echo "[info] verifying image runs locally ..."
container_id=$(docker run -d --rm -p 8000:8000 "household_basket_env:${IMAGE_TAG}")
trap 'docker stop "${container_id}" >/dev/null 2>&1 || true' EXIT
sleep 4
if ! curl -fs http://localhost:8000/health >/dev/null; then
  echo "[error] health check failed" >&2
  exit 1
fi
echo "[info] health OK; pushing to HuggingFace Space ..."
docker stop "${container_id}" >/dev/null

openenv push \
  --image "household_basket_env:${IMAGE_TAG}" \
  --hf-space "${HF_USERNAME}/${HF_SPACE_NAME}" \
  ${HF_TOKEN:+--hf-token "${HF_TOKEN}"}

echo "[done] pushed to https://huggingface.co/spaces/${HF_USERNAME}/${HF_SPACE_NAME}"
