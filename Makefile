# Absolute sqlite store so `train` and `ui` always read/write the same place,
# regardless of the directory make is invoked from.
TRACKING_URI := sqlite:///$(abspath mlflow.db)
UI_PORT := 5001

# Extra flags for a run, e.g. `make train ARGS="--epochs 30 --backbone-lr 1e-4"`.
ARGS ?=

.PHONY: train ui

train:
	uv run python scripts/training.py $(ARGS)

ui:
	uv run mlflow ui --port $(UI_PORT) --backend-store-uri $(TRACKING_URI)
