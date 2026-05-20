"""Named prompt strategies for LLM student simulation.

Each strategy defines how the prompt is assembled from available components:
persona, raw examples, RAG context, summarized history.

Usage:
    python -m llm_simulator.run --models claude --strategy v4_summarized ...

Strategies:
    v1_baseline:     Simple student profile + raw code examples in a single
                     user message. No system message. This is the original
                     GLM prompt format.

    v2_persona:      Rich persona (system message) + raw code examples
                     (user message). Persona includes behavioral archetype,
                     per-topic knowledge state, coding style indicators.

    v3_persona_rag:  Rich persona (system message) + raw code examples +
                     RAG-retrieved student prior work on other questions
                     (user message).

    v4_summarized:   Rich persona (system message) + Haiku-summarized
                     behavioral descriptions of prior work (user message).
                     No raw code in examples. No RAG (redundant with
                     summarized history).

CLI flags map to strategies:
    (no flags)                    -> v1_baseline
    --persona                     -> v2_persona
    --persona --rag               -> v3_persona_rag
    --persona --summarize         -> v4_summarized
"""

from dataclasses import dataclass


@dataclass
class PromptStrategy:
    name: str
    description: str
    use_persona: bool
    use_raw_examples: bool
    use_rag: bool
    use_summarized_history: bool

    @property
    def cli_flags(self) -> str:
        flags = []
        if self.use_persona:
            flags.append("--persona")
        if self.use_rag:
            flags.append("--rag")
        if self.use_summarized_history:
            flags.append("--summarize")
        return " ".join(flags) if flags else "(no flags)"


STRATEGIES = {
    "v1_baseline": PromptStrategy(
        name="v1_baseline",
        description=(
            "Simple student profile + raw code examples in a single user message. "
            "No system message. Original GLM prompt format."
        ),
        use_persona=False,
        use_raw_examples=True,
        use_rag=False,
        use_summarized_history=False,
    ),
    "v2_persona": PromptStrategy(
        name="v2_persona",
        description=(
            "Rich persona as system message (behavioral archetype, per-topic "
            "knowledge state, coding style). Raw code examples in user message."
        ),
        use_persona=True,
        use_raw_examples=True,
        use_rag=False,
        use_summarized_history=False,
    ),
    "v3_persona_rag": PromptStrategy(
        name="v3_persona_rag",
        description=(
            "Rich persona as system message + raw code examples + RAG-retrieved "
            "student prior work on other questions in user message."
        ),
        use_persona=True,
        use_raw_examples=True,
        use_rag=True,
        use_summarized_history=False,
    ),
    "v4_summarized": PromptStrategy(
        name="v4_summarized",
        description=(
            "Rich persona as system message + Haiku-summarized behavioral "
            "descriptions of prior work in user message. No raw code in "
            "examples. No RAG (redundant with summarized history)."
        ),
        use_persona=True,
        use_raw_examples=False,
        use_rag=False,
        use_summarized_history=True,
    ),
}


def get_strategy(persona: bool, rag: bool, summarize: bool) -> PromptStrategy:
    if summarize:
        return STRATEGIES["v4_summarized"]
    if rag and persona:
        return STRATEGIES["v3_persona_rag"]
    if persona:
        return STRATEGIES["v2_persona"]
    return STRATEGIES["v1_baseline"]


def print_strategies():
    for name, s in STRATEGIES.items():
        print(f"{name}: {s.description}")
        print(f"  CLI: {s.cli_flags}")
        print()
