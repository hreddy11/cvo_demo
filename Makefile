.PHONY: sync data features train test up down logs rebuild status
sync:     ; uv sync
data:     ; uv run python scripts/download_data.py
features: ; uv run python scripts/build_features.py
train:    ; uv run python scripts/train_model.py
test:     ; uv run pytest -q
up:       ; docker compose up -d --build && docker compose ps
down:     ; docker compose down
logs:     ; docker compose logs -f cvo-api
rebuild:  ; docker compose build --no-cache
status:   ; curl -s localhost:8000/model | python3 -m json.tool | head -20
