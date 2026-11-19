"""Probe questions (PQs) and the per-judge probe sets used in the paper.

`probe_questions.yaml` is the paper's 20-probe corpus (Appendix A, Table 1),
copied unchanged from the reference implementation
(https://github.com/freeze-lasr/aware_bench, MIT licence, commit
77e3cb91d5937204f43da6b4cd36480739bd7e91, `probe_questions/eval_mcq_calibration.yaml`).
"""

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

PROBE_FILE = Path(__file__).parent / "probe_questions.yaml"


class ProbeAnswer(BaseModel):
    """The two labels a probe accepts. Which label means "evaluation" varies by probe."""

    real: str
    eval: str


class Probe(BaseModel):
    """One closed-ended probe question appended after a transcript."""

    model_config = ConfigDict(extra="ignore")

    id: str
    category: str
    text: str
    template_text: str
    answer: ProbeAnswer

    @property
    def label_values(self) -> dict[str, float]:
        """Map each accepted label to its P(eval) vote: 1.0 = evaluation, 0.0 = deployment."""
        return {self.answer.real: 0.0, self.answer.eval: 1.0}


@cache
def load_probes() -> dict[str, Probe]:
    """Load the 20-probe corpus, keyed by probe id, in file (paper table) order."""
    with PROBE_FILE.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    probes = [Probe.model_validate(x) for x in raw if not x.get("deprecated")]
    return {p.id: p for p in probes}


# The probe set each judge used for the paper's headline AUROCs (Figure 6).
# Read from the published measurement scores
# (huggingface.co/datasets/el7982/aware-bench, measure_logs/scores.csv,
# revision 2a036d1a642b64c6aea8b681a47f2624b80fd16d). Each set is the
# exhaustive best subset of size <= 5 on that judge's 20% calibration split.
# These differ from the sets printed in the paper's Table 2, which come from a
# separate six-judge selection run.
PAPER_PROBE_SETS: dict[str, list[str]] = {
    "claude-haiku-4-5": [
        "h_authenticity_a",
        "h_outcome_a",
        "h_purpose_function_a",
        "h_stakeholder_a",
        "indirect_organic_flow_a",
    ],
    "claude-sonnet-4-6": [
        "h_motivation_a",
        "h_origin_a",
        "h_purpose_function_a",
        "indirect_organic_flow_a",
    ],
    "claude-opus-4-6": [
        "h_intent_and_style_a",
        "h_lifecycle_a",
        "h_naturalness_a",
        "h_purpose_function_a",
        "indirect_requester_knowledge_a",
    ],
    "gpt-4.1": [
        "h_authenticity_a",
        "h_naturalness_a",
        "h_outcome_a",
        "h_provenance_a",
        "h_purpose_function_a",
    ],
    "gpt-5.1": ["h_intent_and_style_a", "h_motivation_a", "indirect_organic_flow_a"],
    "gpt-5.4": [
        "h_audience_a",
        "h_intent_and_style_a",
        "indirect_organic_flow_a",
        "indirect_requester_knowledge_a",
    ],
    "gemini-2.5-pro": [
        "h_audience_a",
        "h_environment_a",
        "h_origin_a",
        "h_purpose_analysis_a",
        "h_purpose_function_a",
    ],
    "gemini-3.1-pro-preview": [
        "direct_vn_a",
        "h_origin_a",
        "h_stakeholder_a",
        "indirect_consequences_a",
        "indirect_organic_flow_a",
    ],
    "deepseek-v3.2": [
        "h_motivation_a",
        "h_origin_a",
        "h_provenance_a",
        "h_stakeholder_a",
        "indirect_organic_flow_a",
    ],
    "glm-5.1": [
        "h_audience_a",
        "h_lifecycle_a",
        "h_outcome_a",
        "h_stakeholder_a",
        "h_value_target_a",
    ],
    "kimi-k2.5": [
        "h_motivation_a",
        "h_outcome_a",
        "h_purpose_function_a",
        "h_stakeholder_a",
        "indirect_organic_flow_a",
    ],
    "qwen3.5-397b-a17b": [
        "h_motivation_a",
        "h_origin_a",
        "h_purpose_utility_a",
        "h_stakeholder_a",
        "indirect_organic_flow_a",
    ],
}

# The paper's Figure 6 AUROC for each judge, recomputed exactly from the same scores.csv.
PAPER_AUROC: dict[str, float] = {
    "claude-haiku-4-5": 0.758,
    "claude-sonnet-4-6": 0.878,
    "claude-opus-4-6": 0.892,
    "gpt-4.1": 0.784,
    "gpt-5.1": 0.802,
    "gpt-5.4": 0.825,
    "gemini-2.5-pro": 0.763,
    "gemini-3.1-pro-preview": 0.872,
    "deepseek-v3.2": 0.796,
    "glm-5.1": 0.779,
    "kimi-k2.5": 0.808,
    "qwen3.5-397b-a17b": 0.811,
}


def resolve_probes(probes: str | list[str], pool: str) -> list[Probe]:
    """Turn the task's `probes` argument into a list of probes.

    Args:
        probes: "paper" for the judge's published set, "all" for the full corpus,
            or a list (or comma-separated string) of probe ids.
        pool: The transcript pool / judge name, used to look up the "paper" set.
    """
    corpus = load_probes()
    if probes == "all":
        return list(corpus.values())
    if probes == "paper":
        if pool not in PAPER_PROBE_SETS:
            raise ValueError(
                f"No published probe set for '{pool}'. Pass probes='all' (e.g. for "
                "calibration) or an explicit list of probe ids."
            )
        ids = PAPER_PROBE_SETS[pool]
    elif isinstance(probes, str):
        ids = [p.strip() for p in probes.split(",") if p.strip()]
    else:
        ids = list(probes)
    unknown = [i for i in ids if i not in corpus]
    if unknown:
        raise ValueError(f"Unknown probe ids {unknown}. Known ids: {list(corpus)}")
    if not ids:
        raise ValueError("At least one probe id is required.")
    return [corpus[i] for i in ids]
