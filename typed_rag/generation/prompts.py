"""Prompt templates used for type-aware generation and aggregation."""

from __future__ import annotations

from typing import Dict

# Aspect-level answer generation prompts keyed by question type.
GENERATION_PROMPTS: Dict[str, str] = {
    "Evidence-based": """You are supporting a retrieval-augmented generation system.
Write a concise, factual answer for the question below using ONLY the provided evidence snippets.
Keep the answer under 120 words and cite snippet numbers in square brackets, e.g., [1], [2].

Question: {question}
Aspect: {aspect}

Evidence snippets:
{snippets}

Answer:""",
    "Comparison": """Summarise how the provided evidence compares the specified items.
Cover the requested comparison axis and mention trade-offs. Limit to 140 words.
Use citations with snippet numbers in square brackets.

Question: {question}
Aspect: {aspect}
Axis: {axis}

Evidence snippets:
{snippets}

Answer:""",
    "Experience": """Provide balanced experiential insights drawn from the evidence.
Highlight advantages, drawbacks, and contextual advice. Stay under 140 words.
Use snippet citations in square brackets.

Question: {question}
Aspect: {aspect}

Evidence snippets:
{snippets}

Answer:""",
    "Reason": """Explain the underlying causes or mechanisms using the supplied evidence.
Structure the answer in 2-3 sentences and cite snippet numbers in square brackets.

Question: {question}
Aspect: {aspect}

Evidence snippets:
{snippets}

Answer:""",
    "Instruction": """Produce actionable instructions drawn from the evidence.
List 3-6 short steps. Cite snippet numbers after each step (e.g., [2]).

Question: {question}
Aspect: {aspect}

Evidence snippets:
{snippets}

Steps:""",
    "Debate": """Summarise the viewpoint requested by the aspect.
Present the strongest arguments from the evidence, keep a neutral tone,
and cite snippet numbers in square brackets. Limit to 120 words.

Question: {question}
Perspective: {aspect}

Evidence snippets:
{snippets}

Answer:""",
}

# Aggregation prompts to produce a final response from aspect answers.
AGGREGATION_PROMPTS: Dict[str, str] = {
    "Evidence-based": """Synthesize the evidence-backed answer below into a single concise paragraph (<=120 words).
Maintain citations already present; do not add new ones.

Question: {question}

Aspect answers:
{aspect_answers}

Final answer:""",
    "Comparison": """Combine the aspect answers into a structured comparison for the question.
Begin with a 1-sentence overview, then provide bullet points for each comparison axis.
Preserve and reuse citations that appear in aspect answers.

Question: {question}

Aspect answers:
{aspect_answers}

Final answer:""",
    "Experience": """Craft a helpful recommendation summary covering the perspectives provided.
Begin with a short thesis sentence and follow with bullet recommendations.
Keep original citations intact.

Question: {question}

Aspect answers:
{aspect_answers}

Final answer:""",
    "Reason": """Explain the causes or mechanisms succinctly.
Present key factors in bullet form after a 1-sentence overview.
Reuse existing citations.

Question: {question}

Aspect answers:
{aspect_answers}

Final answer:""",
    "Instruction": """Merge the step lists into a single ordered plan (numbered steps).
If multiple steps overlap, merge them logically. Keep citations after each step.

Question: {question}

Aspect answers:
{aspect_answers}

Final answer:""",
    "Debate": """Adopt a neutral moderator voice. Summarise the debate question by:
1) stating the topic, 2) presenting pro and con summaries, and 3) offering a balanced synthesis.
Respect existing citations.

Question: {question}

Aspect answers:
{aspect_answers}

Final answer:""",
}

