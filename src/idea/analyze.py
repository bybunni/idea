import json

SYSTEM = """\
You are a senior research scientist analyzing an automated experiment.

Determine:
1. VERDICT: positive (idea outperformed), negative (no advantage), or inconclusive
2. ANALYSIS: 2-4 paragraphs on what was tested, results, why, and caveats
3. HEADLINE: one-sentence summary
4. 0-3 NEXT STEPS: concrete follow-up research questions for child loops
   Generate 0 if the result is definitive."""

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["positive", "negative", "inconclusive"]},
        "analysis": {"type": "string"},
        "headline": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["verdict", "analysis", "headline", "next_steps"],
}

SYNTH_SYSTEM = """\
You are synthesizing a parent investigation with its child investigations.
Update the analysis to incorporate child findings."""


def analyze(llm, hypothesis_block, experiment, context=""):
    user = (
        f"## Hypothesis\n{hypothesis_block}\n\n"
        f"## Experiment Results\n{json.dumps(experiment, indent=2)}"
    )
    if context:
        user += f"\n\n## Context\n{context}"
    return llm.structured(SYSTEM, user, SCHEMA, temperature=0.5)


def synthesize(llm, parent, children):
    user = (
        f"## Parent analysis\n{json.dumps(parent, indent=2)}\n\n"
        f"## Child investigations\n{json.dumps(children, indent=2)}"
    )
    return llm.structured(SYNTH_SYSTEM, user, SCHEMA, temperature=0.5)
