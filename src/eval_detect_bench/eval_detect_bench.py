"""EvalDetectBench: can a model tell evaluation transcripts from deployment transcripts?

Li, Ochwang'i, Bharadwaj, Souly and Kirk (2026). "EvalDetectBench: A Benchmark
for Measuring Evaluation Awareness in Frontier Language Models."
Reference implementation: https://github.com/freeze-lasr/aware_bench
"""

from inspect_ai import Task, task
from inspect_ai.model import get_model
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from eval_detect_bench.dataset import (
    TRANSCRIPT_POOLS,
    Split,
    build_dataset,
    download_pool,
    pool_for_model,
)
from eval_detect_bench.probes import resolve_probes
from eval_detect_bench.scorer import probe_label_scorer
from eval_detect_bench.solver import inject_probe

# The version comes from eval.yaml, the file beside this module.
EVAL_VERSION = load_eval_version(__file__)


def resolve_pool(transcripts: str | None) -> str:
    """Pick the transcript pool: the one named, or the one matching the judge model."""
    if transcripts is not None:
        if transcripts not in TRANSCRIPT_POOLS:
            raise ValueError(f"transcripts must be one of {TRANSCRIPT_POOLS}, got '{transcripts}'.")
        return transcripts
    # The judge picks its own transcript pool, as in the paper, so the model is
    # needed when the task is built.
    try:
        judge = get_model()  # inspect-evals-lint: ignore[get_model_location] -- pool follows judge
        model_name = str(judge)
    except ValueError as e:
        raise ValueError(
            f"No judge model is active to pick a transcript pool from. Pass "
            f"transcripts=<pool>, one of {TRANSCRIPT_POOLS}."
        ) from e
    pool = pool_for_model(model_name)
    if pool is None:
        raise ValueError(
            f"No transcript pool matches the model '{model_name}'. Pass "
            f"-T transcripts=<pool>, one of {TRANSCRIPT_POOLS}."
        )
    return pool


@task
def eval_detect_bench(
    transcripts: str | None = None,
    probes: str | list[str] = "paper",
    split: Split = "all",
    calibration_fraction: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
    claude_placeholder: bool | None = None,
) -> Task:
    """Ask the judge model whether each transcript came from an evaluation or from deployment.

    Each sample pairs one transcript with one probe question. A transcript's
    P(eval) is the mean of its probe votes; the headline metric is AUROC of
    P(eval) over evaluation vs deployment transcripts.

    Args:
        transcripts: Transcript pool to judge, named after the model that
            generated its evaluation trajectories (e.g. "claude-haiku-4-5").
            Defaults to the pool matching the judge model, as in the paper.
        probes: "paper" for the probe set the paper selected for this pool's
            judge, "all" for the full 20-probe corpus, or a list (or
            comma-separated string) of probe ids.
        split: "all" scores every transcript, as the paper did. "calibration"
            and "measurement" give the stratified calibration slice used for
            probe selection and its complement.
        calibration_fraction: Fraction of transcripts in the calibration slice.
        seed: Seed for the calibration split and the sample shuffle.
        shuffle: Shuffle samples so `--limit` draws from every source.
        claude_placeholder: Whether to add the reference's placeholder assistant
            turn when a transcript ends on a user message. None follows the
            reference: on for model names starting with "claude" (the direct
            Anthropic API), off otherwise, including Claude via OpenRouter. Set
            True to reproduce the paper's Claude Haiku 4.5 prompts through OpenRouter.

    Returns:
        The EvalDetectBench task.
    """
    pool = resolve_pool(transcripts)
    probe_list = resolve_probes(probes, pool)
    dataset = build_dataset(
        download_pool(pool),
        probe_list,
        split=split,
        calibration_fraction=calibration_fraction,
        seed=seed,
        shuffle=shuffle,
        name=f"eval_detect_bench_{pool}",
    )
    return Task(
        dataset=dataset,
        solver=[inject_probe(claude_placeholder), generate()],
        scorer=probe_label_scorer(),
        metadata={"transcript_pool": pool, "probe_ids": [p.id for p in probe_list]},
        version=EVAL_VERSION,
    )
