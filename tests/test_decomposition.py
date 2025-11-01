#!/usr/bin/env python3
"""
Unit tests for Typed-RAG Query Decomposition

Tests decomposition for all 6 question types with 5 examples each.
"""

import unittest
from pathlib import Path
from typed_rag.classifier import QuestionType, classify_question
from typed_rag.decompose import QueryDecomposer, SubQuery


class TestQueryDecomposition(unittest.TestCase):
    """Test query decomposition for all question types."""

    @classmethod
    def setUpClass(cls):
        """Set up decomposer once for all tests."""
        cls.decomposer = QueryDecomposer(cache_dir=Path("./test_cache"))

    def _test_decomposition(self, question: str, expected_type: str, min_subqueries: int, max_subqueries: int):
        """Helper to test decomposition."""
        # Classify
        qtype = classify_question(question, use_llm=False)

        # Decompose
        plan = self.decomposer.decompose(question, expected_type, use_cache=False)

        # Assertions
        self.assertEqual(plan.question_type, expected_type)
        self.assertEqual(plan.original_question, question)
        self.assertGreaterEqual(len(plan.sub_queries), min_subqueries)
        self.assertLessEqual(len(plan.sub_queries), max_subqueries)

        # Check all sub-queries have required fields
        for sq in plan.sub_queries:
            self.assertIsInstance(sq, SubQuery)
            self.assertIsNotNone(sq.aspect)
            self.assertIsNotNone(sq.query)
            self.assertIsNotNone(sq.strategy)
            self.assertGreater(len(sq.query), 0)

        return plan

    # ========================================================================
    # Evidence-based Tests (1 sub-query expected)
    # ========================================================================

    def test_evidence_1(self):
        """Test: What is quantum computing?"""
        plan = self._test_decomposition(
            "What is quantum computing?",
            QuestionType.EVIDENCE,
            1, 1
        )
        self.assertEqual(plan.sub_queries[0].strategy, "evidence")

    def test_evidence_2(self):
        """Test: Explain photosynthesis"""
        plan = self._test_decomposition(
            "Explain photosynthesis",
            QuestionType.EVIDENCE,
            1, 1
        )
        self.assertEqual(plan.sub_queries[0].strategy, "evidence")

    def test_evidence_3(self):
        """Test: What is machine learning?"""
        plan = self._test_decomposition(
            "What is machine learning?",
            QuestionType.EVIDENCE,
            1, 1
        )
        self.assertEqual(plan.sub_queries[0].strategy, "evidence")

    def test_evidence_4(self):
        """Test: Define blockchain technology"""
        plan = self._test_decomposition(
            "Define blockchain technology",
            QuestionType.EVIDENCE,
            1, 1
        )
        self.assertEqual(plan.sub_queries[0].strategy, "evidence")

    def test_evidence_5(self):
        """Test: What are neural networks?"""
        plan = self._test_decomposition(
            "What are neural networks?",
            QuestionType.EVIDENCE,
            1, 1
        )
        self.assertEqual(plan.sub_queries[0].strategy, "evidence")

    # ========================================================================
    # Comparison Tests (3-5 sub-queries expected)
    # ========================================================================

    def test_comparison_1(self):
        """Test: Python vs Java for web development"""
        plan = self._test_decomposition(
            "Python vs Java for web development",
            QuestionType.COMPARISON,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "compare")

    def test_comparison_2(self):
        """Test: React vs Vue.js comparison"""
        plan = self._test_decomposition(
            "React vs Vue.js comparison",
            QuestionType.COMPARISON,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "compare")

    def test_comparison_3(self):
        """Test: Difference between SQL and NoSQL databases"""
        plan = self._test_decomposition(
            "Difference between SQL and NoSQL databases",
            QuestionType.COMPARISON,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "compare")

    def test_comparison_4(self):
        """Test: Which is better: iPhone or Android?"""
        plan = self._test_decomposition(
            "Which is better: iPhone or Android?",
            QuestionType.COMPARISON,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "compare")

    def test_comparison_5(self):
        """Test: Docker vs Kubernetes comparison"""
        plan = self._test_decomposition(
            "Docker vs Kubernetes comparison",
            QuestionType.COMPARISON,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "compare")

    # ========================================================================
    # Experience Tests (2-4 sub-queries expected)
    # ========================================================================

    def test_experience_1(self):
        """Test: Should I invest in stocks or bonds?"""
        plan = self._test_decomposition(
            "Should I invest in stocks or bonds?",
            QuestionType.EXPERIENCE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "opinion")

    def test_experience_2(self):
        """Test: Is it worth buying a MacBook Pro?"""
        plan = self._test_decomposition(
            "Is it worth buying a MacBook Pro?",
            QuestionType.EXPERIENCE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "opinion")

    def test_experience_3(self):
        """Test: Should I learn React or Angular?"""
        plan = self._test_decomposition(
            "Should I learn React or Angular?",
            QuestionType.EXPERIENCE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "opinion")

    def test_experience_4(self):
        """Test: Is cloud computing worth the investment?"""
        plan = self._test_decomposition(
            "Is cloud computing worth the investment?",
            QuestionType.EXPERIENCE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "opinion")

    def test_experience_5(self):
        """Test: Should I use TypeScript or JavaScript?"""
        plan = self._test_decomposition(
            "Should I use TypeScript or JavaScript?",
            QuestionType.EXPERIENCE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "opinion")

    # ========================================================================
    # Reason Tests (2-4 sub-queries expected)
    # ========================================================================

    def test_reason_1(self):
        """Test: Why is the sky blue?"""
        plan = self._test_decomposition(
            "Why is the sky blue?",
            QuestionType.REASON,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "evidence")

    def test_reason_2(self):
        """Test: What causes climate change?"""
        plan = self._test_decomposition(
            "What causes climate change?",
            QuestionType.REASON,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "evidence")

    def test_reason_3(self):
        """Test: Why do we need sleep?"""
        plan = self._test_decomposition(
            "Why do we need sleep?",
            QuestionType.REASON,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "evidence")

    def test_reason_4(self):
        """Test: Explain why Python is popular"""
        plan = self._test_decomposition(
            "Explain why Python is popular",
            QuestionType.REASON,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "evidence")

    def test_reason_5(self):
        """Test: What is the reason for inflation?"""
        plan = self._test_decomposition(
            "What is the reason for inflation?",
            QuestionType.REASON,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "evidence")

    # ========================================================================
    # Instruction Tests (3-5 sub-queries expected)
    # ========================================================================

    def test_instruction_1(self):
        """Test: How to install Python on Mac?"""
        plan = self._test_decomposition(
            "How to install Python on Mac?",
            QuestionType.INSTRUCTION,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "steps")

    def test_instruction_2(self):
        """Test: How do I create a React app?"""
        plan = self._test_decomposition(
            "How do I create a React app?",
            QuestionType.INSTRUCTION,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "steps")

    def test_instruction_3(self):
        """Test: Steps to deploy to AWS"""
        plan = self._test_decomposition(
            "Steps to deploy to AWS",
            QuestionType.INSTRUCTION,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "steps")

    def test_instruction_4(self):
        """Test: How to build a REST API?"""
        plan = self._test_decomposition(
            "How to build a REST API?",
            QuestionType.INSTRUCTION,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "steps")

    def test_instruction_5(self):
        """Test: Guide to setting up Docker"""
        plan = self._test_decomposition(
            "Guide to setting up Docker",
            QuestionType.INSTRUCTION,
            3, 5
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "steps")

    # ========================================================================
    # Debate Tests (2-4 sub-queries expected, including synthesis)
    # ========================================================================

    def test_debate_1(self):
        """Test: Pros and cons of remote work"""
        plan = self._test_decomposition(
            "Pros and cons of remote work",
            QuestionType.DEBATE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "stance")

    def test_debate_2(self):
        """Test: Arguments for and against AI regulation"""
        plan = self._test_decomposition(
            "Arguments for and against AI regulation",
            QuestionType.DEBATE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "stance")

    def test_debate_3(self):
        """Test: Controversial aspects of cryptocurrency"""
        plan = self._test_decomposition(
            "Controversial aspects of cryptocurrency",
            QuestionType.DEBATE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "stance")

    def test_debate_4(self):
        """Test: Advantages and disadvantages of nuclear energy"""
        plan = self._test_decomposition(
            "Advantages and disadvantages of nuclear energy",
            QuestionType.DEBATE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "stance")

    def test_debate_5(self):
        """Test: Different perspectives on social media impact"""
        plan = self._test_decomposition(
            "Different perspectives on social media impact",
            QuestionType.DEBATE,
            2, 4
        )
        for sq in plan.sub_queries:
            self.assertEqual(sq.strategy, "stance")


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQueryDecomposition)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return result


if __name__ == "__main__":
    run_tests()

