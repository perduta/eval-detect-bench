"""Per-judge probe selection from a calibration run (paper Section 3.2.2).

Run the task with `-T split=calibration -T probes=all`, then:

    uv run python -m eval_detect_bench.selection logs/<calibration-log>.eval

This prints the probe ids to pass as `-T probes=...` for the measurement run.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from inspect_ai.log import EvalLog, read_eval_log

from eval_detect_bench.metrics import PARSE_FAILURE, roc_auc

SCORER_NAME = "probe_label_scorer"


@dataclass
class ProbeSelection:
    probe_ids: list[str]
    calibration_auroc: float
    n_transcripts: int


def vote_matrix(log: EvalLog) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Pivot a calibration log into a (transcripts x probes) vote matrix.

    Parse failures are NaN. Returns (votes, probe_ids sorted, labels).
    """
    votes: dict[str, dict[str, float]] = defaultdict(dict)
    label: dict[str, int] = {}
    for sample in log.samples or []:
        score = (sample.scores or {}).get(SCORER_NAME)
        if score is None:
            continue
        value = score.as_float()
        if value == PARSE_FAILURE:
            continue
        tid = str(sample.metadata["transcript_id"])
        votes[tid][str(sample.metadata["probe_id"])] = value
        label[tid] = int(bool(sample.metadata["eval_category"]))

    probe_ids = sorted({p for row in votes.values() for p in row})
    tids = sorted(votes)
    matrix = np.full((len(tids), len(probe_ids)), np.nan)
    col = {p: j for j, p in enumerate(probe_ids)}
    for i, tid in enumerate(tids):
        for p, v in votes[tid].items():
            matrix[i, col[p]] = v
    labels = np.array([label[t] for t in tids], dtype=int)
    return matrix, probe_ids, labels


def ensemble_auroc(votes: np.ndarray, labels: np.ndarray, cols: list[int]) -> float | None:
    """AUROC of the mean vote over `cols`, using transcripts with no missing vote there."""
    subset = votes[:, cols]
    valid = ~np.isnan(subset).any(axis=1)
    if valid.sum() < 2:
        return None
    auc = roc_auc(labels[valid], subset[valid].mean(axis=1))
    return None if np.isnan(auc) else auc


def select_exhaustive(
    votes: np.ndarray, probe_ids: list[str], labels: np.ndarray, max_k: int = 5
) -> ProbeSelection:
    """Try every probe subset of size 1..max_k and keep the one with the best AUROC.

    Follows the reference: probes with zero variance are dropped first, subsets
    are tried in order of size and then lexicographic order, and a later subset
    replaces the best only if its AUROC is strictly higher.
    """
    if len(probe_ids) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("Calibration data needs probes and both eval and deploy transcripts.")

    keep = []
    for j in range(len(probe_ids)):
        col = votes[:, j]
        col = col[~np.isnan(col)]
        if len(col) > 1 and np.std(col) > 1e-9:
            keep.append(j)
    if not keep:
        raise ValueError("Every probe gave the same answer on all calibration transcripts.")
    votes = votes[:, keep]
    probe_ids = [probe_ids[j] for j in keep]

    best_auc = -1.0
    best: tuple[int, ...] | None = None
    for k in range(1, min(max_k, len(probe_ids)) + 1):
        for combo in combinations(range(len(probe_ids)), k):
            auc = ensemble_auroc(votes, labels, list(combo))
            if auc is not None and auc > best_auc:
                best_auc, best = auc, combo
    if best is None:
        raise ValueError("No probe subset produced a defined AUROC.")
    return ProbeSelection(
        probe_ids=[probe_ids[i] for i in best],
        calibration_auroc=best_auc,
        n_transcripts=len(labels),
    )


def select_probes(log: str | Path | EvalLog, max_k: int = 5) -> ProbeSelection:
    """Select a judge's probe set from a calibration log."""
    if not isinstance(log, EvalLog):
        log = read_eval_log(str(log))
    votes, probe_ids, labels = vote_matrix(log)
    return select_exhaustive(votes, probe_ids, labels, max_k=max_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a judge's probe set from a calibration log."
    )
    parser.add_argument("log", help="Calibration .eval log (run with probes=all).")
    parser.add_argument("--max-k", type=int, default=5, help="Largest probe-set size to try.")
    args = parser.parse_args()
    selection = select_probes(args.log, max_k=args.max_k)
    print(f"Calibration transcripts: {selection.n_transcripts}")
    print(f"Calibration AUROC:      {selection.calibration_auroc:.4f}")
    print(f"Selected probes:        {','.join(selection.probe_ids)}")


if __name__ == "__main__":
    main()
