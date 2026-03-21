#!/usr/bin/env bash

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
HEALTH_URL="${HEALTH_URL:-}"
HEALTH_RETRIES="${HEALTH_RETRIES:-20}"
HEALTH_INTERVAL_SECONDS="${HEALTH_INTERVAL_SECONDS:-3}"

if [ -z "$DEPLOY_DIR" ]; then
  echo "DEPLOY_DIR is required"
  exit 1
fi

if [ ! -d "$DEPLOY_DIR" ]; then
  echo "DEPLOY_DIR does not exist: $DEPLOY_DIR"
  exit 1
fi

if [[ "$COMPOSE_FILE" = /* ]]; then
  compose_path="$COMPOSE_FILE"
else
  compose_path="$DEPLOY_DIR/$COMPOSE_FILE"
fi

if [ ! -f "$compose_path" ]; then
  echo "Compose file not found: $compose_path"
  exit 1
fi

cd "$DEPLOY_DIR"

echo "Deploy directory: $DEPLOY_DIR"
echo "Compose file: $compose_path"

docker compose -f "$compose_path" pull
docker compose -f "$compose_path" up -d
docker compose -f "$compose_path" ps

if [ -z "$HEALTH_URL" ]; then
  echo "HEALTH_URL is empty, skipping health check"
  exit 0
fi

echo "Checking health: $HEALTH_URL"
for attempt in $(seq 1 "$HEALTH_RETRIES"); do
  if curl -fsS -m 5 "$HEALTH_URL"; then
    echo
    echo "Health check passed on attempt ${attempt}"
    exit 0
  fi

  if [ "$attempt" -lt "$HEALTH_RETRIES" ]; then
    sleep "$HEALTH_INTERVAL_SECONDS"
  fi
done

echo "Health check failed after ${HEALTH_RETRIES} attempts"
docker compose -f "$compose_path" ps
exit 1
