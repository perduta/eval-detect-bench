"""Load EvalDetectBench transcripts and pair each one with each probe question."""

import json
import random
import re
from collections import defaultdict
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_evals_utils.huggingface import hf_hub_download

from eval_detect_bench.probes import Probe

HF_REPO_ID = "el7982/aware-bench"
# Pinned dataset commit. Bump the task version in eval.yaml if this changes.
HF_REVISION = "2a036d1a642b64c6aea8b681a47f2624b80fd16d"

# One transcript pool per judge in the paper. Each pool holds that model's own
# evaluation trajectories plus the shared deployment transcripts.
TRANSCRIPT_POOLS: tuple[str, ...] = (
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "gpt-4.1",
    "gpt-5.1",
    "gpt-5.4",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
    "deepseek-v3.2",
    "glm-5.1",
    "kimi-k2.5",
    "qwen3.5-397b-a17b",
)

# Metadata kept from each trajectory record. Source-benchmark fields (graders,
# hidden targets and so on) are dropped so they don't bloat the logs.
KEPT_METADATA = (
    "category",
    "eval_category",
    "has_tools",
    "subcategory",
    "scaffold",
    "generator_model",
    "generator_family",
    "source_split",
    "tools",
)

# The reference pipeline's calibration split settings (configs/reproduce.yaml).
STRATIFY_BY = ("eval_category", "has_tools")

Split = Literal["all", "calibration", "measurement"]


def pool_for_model(model_name: str) -> str | None:
    """Return the transcript pool matching a model name, if there is one.

    Matches on the last path segment, so `openrouter/openai/gpt-5.1` and
    `openai/gpt-5.1` both map to `gpt-5.1`. Dots and hyphens in version numbers
    are treated alike, so OpenRouter's `claude-haiku-4.5` maps to
    `claude-haiku-4-5`. A date suffix is allowed (`claude-haiku-4-5-20251001`,
    `gpt-4.1-2025-04-14`), but other suffixes are different models:
    `gpt-4.1-mini` has no pool. OpenRouter's dated Claude names put the version
    first (`claude-4.5-haiku-20251001`) and are reordered before matching.
    """
    base = model_name.rsplit("/", 1)[-1].replace(".", "-")
    base = re.sub(r"^claude-(\d+(?:-\d+)?)-(haiku|sonnet|opus)(?=-\d{8}$|$)", r"claude-\2-\1", base)
    for pool in TRANSCRIPT_POOLS:
        if re.fullmatch(re.escape(pool.replace(".", "-")) + r"(-\d{8}|-\d{4}-\d{2}-\d{2})?", base):
            return pool
    return None


def download_pool(pool: str) -> list[dict[str, Any]]:
    """Download a pool's `all.json` at the pinned revision and return its records."""
    if pool not in TRANSCRIPT_POOLS:
        raise ValueError(f"Unknown transcript pool '{pool}'. Choose one of {TRANSCRIPT_POOLS}.")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"collected_trajectories/{pool}/all.json",
        revision=HF_REVISION,
    )
    with open(path, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    return records


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert one trajectory record into a transcript-level sample."""
    meta = record["metadata"]
    metadata = {k: meta[k] for k in KEPT_METADATA if k in meta}
    metadata["eval_category"] = bool(meta["eval_category"])
    metadata["has_tools"] = bool(meta.get("has_tools", False))
    metadata.setdefault("tools", [])
    return Sample(id=record["id"], input=record["input"], metadata=metadata)


def stratified_split(
    samples: list[Sample],
    calibration_fraction: float,
    seed: int,
    stratify_by: tuple[str, ...] = STRATIFY_BY,
) -> tuple[list[Sample], list[Sample]]:
    """Split transcripts into disjoint calibration and measurement sets.

    A line-for-line port of the reference `stratified_split`, so the same input
    order and seed give the same calibration set.
    """
    if calibration_fraction <= 0.0:
        return [], list(samples)
    if calibration_fraction >= 1.0:
        return list(samples), []

    groups: dict[tuple[Any, ...], list[Sample]] = defaultdict(list)
    for s in samples:
        meta = s.metadata or {}
        groups[tuple(meta.get(k) for k in stratify_by)].append(s)

    rng = random.Random(seed)
    calibration: list[Sample] = []
    measurement: list[Sample] = []
    for key in sorted(groups, key=str):
        group = list(groups[key])
        rng.shuffle(group)
        n_cal = max(1, int(len(group) * calibration_fraction))
        calibration.extend(group[:n_cal])
        measurement.extend(group[n_cal:])
    return calibration, measurement


def pair_with_probe(transcript: Sample, probe: Probe) -> Sample:
    """Make the (transcript, probe) sample the judge actually answers."""
    assert transcript.id is not None
    metadata = dict(transcript.metadata or {})
    metadata.update(transcript_id=str(transcript.id), probe_id=probe.id)
    return Sample(
        id=f"{transcript.id}:{probe.id}",
        input=transcript.input,
        target=json.dumps(probe.label_values),
        metadata=metadata,
    )


def build_dataset(
    records: list[dict[str, Any]],
    probes: list[Probe],
    split: Split = "all",
    calibration_fraction: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
    name: str = "eval_detect_bench",
) -> MemoryDataset:
    """Build the transcript x probe dataset for one pool."""
    transcripts = [record_to_sample(r) for r in records]
    if split != "all":
        calibration, measurement = stratified_split(transcripts, calibration_fraction, seed)
        transcripts = calibration if split == "calibration" else measurement
    samples = [pair_with_probe(t, p) for t in transcripts for p in probes]
    if shuffle:
        # Same shuffle as the reference, so `--limit` draws a mix of sources.
        random.Random(seed).shuffle(samples)
    return MemoryDataset(samples, name=name)
