"""Move the post-deduplication runs into their own MLflow experiment.

The leaky-split runs (LEAKY-EXPERIMENTS.md) and the clean-data runs
(EXPERIMENTS.md) are not comparable - same metric name, different meaning - so
keeping them in one experiment invites exactly the mistake the cleanup was for.
This moves the clean runs to `pokemon-classification-clean` and renames them to
the `c0-*` scheme EXPERIMENTS.md uses.

    uv run python scripts/migrate_clean_runs.py [--dry-run]

MLflow exposes no API for reassigning a run's experiment, so this updates the
sqlite backing store directly. It only rewrites `experiment_id`; `artifact_uri`
is stored per-run as an absolute path and keeps resolving, so artifacts stay
readable from their original location on disk rather than being copied.
"""

import argparse
import sqlite3
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
CLEAN_EXPERIMENT = "pokemon-classification-clean"

# Runs made after the shiny-sprite filter landed, with the names EXPERIMENTS.md
# refers to them by.
RENAMES = {
    "p12-random-baseline": "c0-leaky-reference",
    "p12-noshiny-seed42": "c0-singlesplit-seed42",
    "p12-noshiny-seed43": "c0-singlesplit-seed43",
    "p12-noshiny-seed44": "c0-singlesplit-seed44",
    "p12-noshiny-grouped-5fold": "c0-5fold",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = mlflow.get_experiment_by_name(CLEAN_EXPERIMENT)
    if experiment is None:
        if args.dry_run:
            print(f"would create experiment {CLEAN_EXPERIMENT}")
            target_id = "<new>"
        else:
            # Pin artifact_location for the same reason scripts/training.py does:
            # MLflow bakes it in at creation time and never updates it.
            target_id = mlflow.create_experiment(
                CLEAN_EXPERIMENT,
                artifact_location=f"file://{PROJECT_ROOT / 'mlruns'}",
            )
            print(f"created experiment {CLEAN_EXPERIMENT} (id {target_id})")
    else:
        target_id = experiment.experiment_id
        print(f"using existing experiment {CLEAN_EXPERIMENT} (id {target_id})")

    source = mlflow.get_experiment_by_name("pokemon-classification")
    runs = client.search_runs([source.experiment_id], max_results=1000)

    # Run names are not unique - an interrupted run leaves a RUNNING record
    # behind that a later rerun does not replace. Selecting by name alone will
    # happily migrate the killed one. Take only FINISHED runs, and refuse to
    # guess if that still leaves more than one.
    by_name = {}
    for run in runs:
        if run.info.status == "FINISHED":
            by_name.setdefault(run.data.tags.get("mlflow.runName"), []).append(run)

    missing = [name for name in RENAMES if name not in by_name]
    if missing:
        print(f"WARNING: no FINISHED run found, skipping: {missing}")

    ambiguous = {name: len(found) for name, found in by_name.items() if len(found) > 1}
    if ambiguous.keys() & RENAMES.keys():
        raise SystemExit(f"ambiguous run names, resolve manually: {ambiguous}")

    moves = [(by_name[old][0], old, new) for old, new in RENAMES.items() if old in by_name]
    for run, old, new in moves:
        print(f"  {old}  ->  {new}  ({run.info.run_id[:8]})")

    if args.dry_run:
        print(f"\ndry run: {len(moves)} runs would move to {CLEAN_EXPERIMENT}")
        return

    for run, _, new in moves:
        client.set_tag(run.info.run_id, "mlflow.runName", new)

    connection = sqlite3.connect(PROJECT_ROOT / "mlflow.db")
    try:
        with connection:
            connection.executemany(
                "UPDATE runs SET experiment_id = ? WHERE run_uuid = ?",
                [(target_id, run.info.run_id) for run, _, _ in moves],
            )
    finally:
        connection.close()

    print(f"\nmoved {len(moves)} runs to {CLEAN_EXPERIMENT}")


if __name__ == "__main__":
    main()
