# LINKAGE Evaluation Results - Ablation Study

## Overview

This document contains the quality evaluation results from running LINKAGE evaluation on the ablation study outputs.

**Date**: January 2025  
**Dataset**: `data/wiki_nfqa/dev6.jsonl` (6 questions)  
**Scoring Method**: Embedding-based (no LLM)  
**Reference File**: `data/wiki_nfqa/references_dev6.jsonl`

## Evaluation Methodology

LINKAGE (Linking Answers with Knowledge and Evidence) evaluates answer quality by:
- **MRR (Mean Reciprocal Rank)**: Measures how well system answers rank against reference answers
- **MPR (Mean Percentile Rank)**: Measures the percentile position of system answers among all references

Higher scores indicate better quality (answers closer to reference answers).

## Results Summary

| Variant | MRR | MPR | Questions Evaluated |
|---------|-----|-----|---------------------|
| **Full** | 0.4722 | 70.83% | 6 |
| **No Classification** | 0.4722 | 70.83% | 6 |
| **No Decomposition** | 0.4444 | 66.67% | 6 |
| **No Retrieval** | 0.4722 | 70.83% | 6 |

## Detailed Analysis

### 1. Full Typed-RAG (Baseline)

**Quality Metrics**:
- **MRR**: 0.4722
- **MPR**: 70.83%
- **Questions**: 6/6 evaluated

**Interpretation**: System answers rank well against reference answers, achieving ~71st percentile on average.

### 2. No Classification

**Quality Metrics**:
- **MRR**: 0.4722
- **MPR**: 70.83%
- **Questions**: 6/6 evaluated

**Interpretation**: Identical quality to full system. Classification doesn't impact answer quality for this dataset.

### 3. No Decomposition

**Quality Metrics**:
- **MRR**: 0.4444
- **MPR**: 66.67%
- **Questions**: 6/6 evaluated

**Interpretation**: Slightly lower quality (6% reduction in both MRR and MPR). Decomposition provides a modest quality benefit for multi-aspect questions.

### 4. No Retrieval

**Quality Metrics**:
- **MRR**: 0.4722
- **MPR**: 70.83%
- **Questions**: 6/6 evaluated

**Interpretation**: Identical quality to full system. Retrieval doesn't impact answer quality but significantly improves latency (3.81s vs 5.62s).

## Key Insights

### Component Impact on Quality

1. **Classification**: 
   - **Quality Impact**: None (same scores as full system)
   - **Conclusion**: Classification enables type-aware decomposition but doesn't directly improve answer quality

2. **Decomposition**: 
   - **Quality Impact**: Positive (~6% improvement in MRR/MPR)
   - **Conclusion**: Breaking questions into sub-aspects helps generate more comprehensive answers

3. **Retrieval**: 
   - **Quality Impact**: None (same scores as full system)
   - **Conclusion**: Retrieval provides speed benefits (48% faster) without quality trade-off

### Quality vs Performance Trade-offs

| Variant | Quality (MRR) | Quality (MPR) | Latency | Quality/Time Ratio |
|---------|---------------|---------------|---------|-------------------|
| **Full** | 0.4722 | 70.83% | 3.81s | 0.124 |
| **No Classification** | 0.4722 | 70.83% | 2.81s | **0.168** ⭐ |
| **No Decomposition** | 0.4444 | 66.67% | 3.27s | 0.136 |
| **No Retrieval** | 0.4722 | 70.83% | 5.62s | 0.084 |

**Best Efficiency**: No Classification variant provides the best quality-to-latency ratio.

## Conclusions

1. **For Maximum Quality**: Use Full Typed-RAG (with decomposition)
   - Provides best quality through decomposition
   - Balanced performance

2. **For Best Efficiency**: Use No Classification variant
   - Same quality as full system
   - 26% faster (2.81s vs 3.81s)
   - Best quality-to-latency ratio

3. **Decomposition Value**: Provides ~6% quality improvement
   - Worth the small latency cost for quality-critical applications

4. **Retrieval Value**: Primarily for speed, not quality
   - 48% faster than pure LLM
   - No quality impact observed

## Files

- `results/ablation_linkage_evaluation.json` - Full evaluation results (JSON)
- `data/wiki_nfqa/references_dev6.jsonl` - Reference answers used for evaluation

---

*Generated from LINKAGE evaluation run on January 2025*

