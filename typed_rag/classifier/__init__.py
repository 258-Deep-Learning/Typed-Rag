"""
Question Type Classifier Module

Provides functionality to classify non-factoid questions into types:
- Evidence-based
- Comparison
- Experience
- Reason
- Instruction
- Debate
"""

from .classifier import (
    QuestionClassifier,
    QuestionType,
    classify_question,
)

__all__ = [
    'QuestionClassifier',
    'QuestionType',
    'classify_question',
]

