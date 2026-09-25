import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "retain_best_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("retain_best_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
metric_by_step = MODULE.metric_by_step
retention_plan = MODULE.retention_plan


def _checkpoint(root: Path, step: int) -> None:
    (root / f"iter_{step:07d}").mkdir()


def test_metric_by_step_extracts_requested_metric(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "prefix eval 19: {'eval/aime': 0.25, 'eval/aime-pass@2': 0.4}\n"
        "prefix eval 39: {'eval/aime': 0.5}\n",
        encoding="utf-8",
    )

    assert metric_by_step(log, "eval/aime") == {19: 0.25, 39: 0.5}


def test_retention_plan_keeps_latest_tied_best_and_final(tmp_path):
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    for step in (19, 39, 59, 199):
        _checkpoint(save_dir, step)
    log = tmp_path / "run.log"
    log.write_text(
        "eval 19: {'eval/aime': 0.4}\n"
        "eval 39: {'eval/aime': 0.6}\n"
        "eval 59: {'eval/aime': 0.6}\n"
        "eval 199: {'eval/aime': 0.5}\n",
        encoding="utf-8",
    )

    assert retention_plan(save_dir, log, "eval/aime") == {
        "metric": "eval/aime",
        "best_step": 59,
        "best_score": 0.6,
        "last_step": 199,
        "retained_steps": [59, 199],
        "removed_steps": [19, 39],
    }
