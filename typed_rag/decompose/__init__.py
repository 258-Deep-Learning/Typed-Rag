"""
Query Decomposition Module

Provides functionality to decompose non-factoid questions into type-aware sub-queries.
"""

from .query_decompose import (
    SubQuery,
    DecompositionPlan,
    QueryDecomposer,
    decompose_question,
)

__all__ = [
    'SubQuery',
    'DecompositionPlan',
    'QueryDecomposer',
    'decompose_question',
]

