#!/usr/bin/env python3
"""
Question Type Classifier for Typed-RAG

Classifies non-factoid questions into one of six types:
- Evidence-based: Questions requiring factual evidence
- Comparison: Questions comparing two or more things
- Experience: Questions about personal experiences or recommendations
- Reason: Questions asking for explanations or causes
- Instruction: Questions asking how to do something
- Debate: Questions about controversial topics or multiple perspectives
"""

import os
import re
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from typed_rag.core.keys import get_fastest_model


class QuestionType:
    """Enumeration of question types."""
    EVIDENCE = "Evidence-based"
    COMPARISON = "Comparison"
    EXPERIENCE = "Experience"
    REASON = "Reason"
    INSTRUCTION = "Instruction"
    DEBATE = "Debate"

    ALL_TYPES = [EVIDENCE, COMPARISON, EXPERIENCE, REASON, INSTRUCTION, DEBATE]


class QuestionClassifier:
    """
    Classifier for non-factoid question types.

    Uses a two-stage approach:
    1. Pattern-based matching for obvious cases (fast)
    2. LLM-based classification for ambiguous cases (accurate)
    """

    def __init__(self, use_llm: bool = True, model_name=None):
        """
        Initialize the classifier.

        Args:
            use_llm: Whether to use LLM for classification (falls back to patterns if False)
            model_name: Gemini model to use for classification
        """
        self.use_llm = use_llm
        self.model_name = get_fastest_model()
        self._llm = None

        # Compile regex patterns for efficiency
        self._patterns = {
            QuestionType.INSTRUCTION: [
                re.compile(r'\bhow\s+to\b', re.IGNORECASE),
                re.compile(r'\bhow\s+do\s+i\b', re.IGNORECASE),
                re.compile(r'\bhow\s+can\s+i\b', re.IGNORECASE),
                re.compile(r'\bsteps\s+to\b', re.IGNORECASE),
                re.compile(r'\bguide\s+to\b', re.IGNORECASE),
            ],
            QuestionType.REASON: [
                re.compile(r'\bwhy\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+causes\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+is\s+the\s+reason\b', re.IGNORECASE),
                re.compile(r'\bexplain\s+why\b', re.IGNORECASE),
            ],
            QuestionType.COMPARISON: [
                re.compile(r'\bversus\b', re.IGNORECASE),
                re.compile(r'\bvs\.?\b', re.IGNORECASE),
                re.compile(r'\bbetter\s+than\b', re.IGNORECASE),
                re.compile(r'\bcompare\b', re.IGNORECASE),
                re.compile(r'\bdifference\s+between\b', re.IGNORECASE),
                re.compile(r'\bwhich\s+is\s+better\b', re.IGNORECASE),
            ],
            QuestionType.EXPERIENCE: [
                re.compile(r'\bshould\s+i\b', re.IGNORECASE),
                re.compile(r'\bis\s+it\s+worth\b', re.IGNORECASE),
                re.compile(r'\brecommend\b', re.IGNORECASE),
                re.compile(r'\bwhat\s+do\s+you\s+think\b', re.IGNORECASE),
                re.compile(r'\byour\s+experience\b', re.IGNORECASE),
            ],
            QuestionType.DEBATE: [
                re.compile(r'\bargue\b', re.IGNORECASE),
                re.compile(r'\bcontroversial\b', re.IGNORECASE),
                re.compile(r'\bpros\s+and\s+cons\b', re.IGNORECASE),
                re.compile(r'\badvantages\s+and\s+disadvantages\b', re.IGNORECASE),
                re.compile(r'\bdebate\b', re.IGNORECASE),
                re.compile(r'\bdifferent\s+perspectives\b', re.IGNORECASE),
            ],
        }

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-load the LLM."""
        if self._llm is None:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set")
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=google_api_key,
                temperature=0.0  # Deterministic for classification
            )
        return self._llm

    def _classify_by_pattern(self, question: str) -> Optional[str]:
        """
        Classify using pattern matching.

        Args:
            question: The question to classify

        Returns:
            Question type if a pattern matches, None otherwise
        """
        question = question.strip()

        # Check patterns in priority order
        for question_type, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(question):
                    return question_type

        return None

    def _classify_by_llm(self, question: str) -> str:
        """
        Classify using LLM (Gemini).

        Args:
            question: The question to classify

        Returns:
            Question type
        """
        prompt = f"""Classify this non-factoid question into exactly one of these categories:
            - Evidence-based: Questions requiring factual evidence or information
            - Comparison: Questions comparing two or more things
            - Experience: Questions about personal experiences, opinions, or recommendations
            - Reason: Questions asking for explanations, causes, or reasons why
            - Instruction: Questions asking how to do something or step-by-step guides
            - Debate: Questions about controversial topics, pros and cons, or multiple perspectives
            
            Question: "{question}"
            
            Return ONLY the category label, nothing else."""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            result = str(response.content).strip()

            # Normalize the response to match our types
            for qtype in QuestionType.ALL_TYPES:
                if qtype.lower() in result.lower():
                    return qtype

            # If no exact match, return Evidence-based as default
            return QuestionType.EVIDENCE

        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")
            return QuestionType.EVIDENCE

    def classify(self, question: str) -> str:
        """
        Classify a question into one of the six types.

        Args:
            question: The question to classify

        Returns:
            Question type (one of QuestionType constants)
        """
        if not question or not question.strip():
            return QuestionType.EVIDENCE

        # Try pattern-based classification first (fast)
        pattern_result = self._classify_by_pattern(question)
        if pattern_result:
            return pattern_result

        # Fall back to LLM if enabled
        if self.use_llm:
            return self._classify_by_llm(question)

        # Default to Evidence-based
        return QuestionType.EVIDENCE

    def classify_with_confidence(self, question: str) -> tuple[str, str]:
        """
        Classify a question and return confidence level.

        Args:
            question: The question to classify

        Returns:
            Tuple of (question_type, confidence) where confidence is "high" or "low"
        """
        pattern_result = self._classify_by_pattern(question)

        if pattern_result:
            # Pattern match has high confidence
            return pattern_result, "high"

        if self.use_llm:
            # LLM classification has medium-high confidence
            llm_result = self._classify_by_llm(question)
            return llm_result, "medium"

        # Default has low confidence
        return QuestionType.EVIDENCE, "low"


def classify_question(question: str, use_llm: bool = True) -> str:
    """
    Convenience function to classify a single question.

    Args:
        question: The question to classify
        use_llm: Whether to use LLM for ambiguous cases

    Returns:
        Question type
    """
    classifier = QuestionClassifier(use_llm=use_llm)
    return classifier.classify(question)


if __name__ == "__main__":
    # Test the classifier
    test_questions = [
        "How to make a cake?",
        "Why is the sky blue?",
        "Python vs Java for web development?",
        "Should I invest in stocks or bonds?",
        "What are the pros and cons of remote work?",
        "What is quantum computing?",
        "How do I install Python on Mac?",
        "What causes climate change?",
        "Is iPhone better than Android?",
        "Are electric cars worth it?",
    ]

    print("Testing Question Classifier\n")
    print("=" * 80)

    classifier = QuestionClassifier(use_llm=True)

    for question in test_questions:
        qtype, confidence = classifier.classify_with_confidence(question)
        print(f"\nQuestion: {question}")
        print(f"Type: {qtype} (confidence: {confidence})")

    print("\n" + "=" * 80)

