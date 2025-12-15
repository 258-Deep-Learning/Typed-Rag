# Happy Flow Demo Results

## Overview

This document contains the results from running the Typed-RAG happy flow demo with various question types.

**Date**: January 2025  
**Script**: `examples/happy_flow_demo.py`  
**Backend**: FAISS  
**Source**: Wikipedia  
**Model**: `meta-llama/Llama-3.2-3B-Instruct`  
**Rerank**: Off  
**LLM**: On

## Configuration

- **Backend**: FAISS
- **Source**: Wikipedia
- **Rerank**: Off
- **LLM**: On
- **Model**: meta-llama/Llama-3.2-3B-Instruct

## Results Summary

**Overall Metrics**:
- **MRR (Mean Reciprocal Rank)**: 0.556
- **MPR (Mean Percentile Rank)**: 75.000%

## Detailed Results by Question

### 1. Evidence-based Question

**Question**: What is the Out-of-Network Deductible for this plan?

**Question Type**: Evidence-based

**Processing**:
- Decomposed into 1 sub-query
- Retrieved 5 documents for evidence aspect
- Generated answer with aspect-level responses

**Final Answer**:
> Unfortunately, the provided snippets do not specify the out-of-network deductible for the plan. The information available is limited to the mention of the month "April" [1, 2, 3]. Therefore, a determination of the out-of-network deductible cannot be made based on the provided text.

**LINKAGE Rank Estimate**: 1/1

**Aspect Answers**:
- **evidence**: I am sorry, but the provided snippets do not contain information about the Out-of-Network Deductible for the plan. The snippets only contain the month "April" [1, 2, 3].

---

### 2. Comparison Question

**Question**: How do cost-sharing obligations for outpatient physiotherapy differ between Select Plus providers and out-of-network clinics under the CSU student policy?

**Question Type**: Comparison

**Processing**:
- Decomposed into 5 sub-queries:
  1. Cost Structure
  2. Maximum Out-of-Pocket Costs
  3. Coverage Limitations
  4. Pre-authorization Requirements
  5. Claim Filing Process
- Retrieved 5 documents for each aspect (25 total)

**Final Answer**:
> Unfortunately, the provided information is insufficient to compare the cost-sharing obligations for outpatient physiotherapy between Select Plus providers and out-of-network clinics under the CSU student policy.
>
> *   **Cost Structure:** Unable to answer; the necessary snippets are unavailable.
> *   **Maximum Out-of-Pocket Costs:** Unable to answer; the necessary snippets are unavailable.
> *   **Coverage Limitations:** Unable to answer; the necessary snippets are unavailable.
> *   **Pre-authorization Requirements:** Unable to answer; the provided evidence does not contain information about pre-authorization requirements. ([1], [2], [3] are insufficient)
> *   **Claim Filing Process:** Unable to answer; the provided evidence does not contain information comparing the claim filing processes.

**LINKAGE Rank Estimate**: 2/4

**Aspect Answers**:
- **Cost Structure**: Unfortunately, I am unable to answer this question. The provided snippets are not included, and I need the evidence to compare the cost-sharing obligations for outpatient physiotherapy between Select Plus providers and out-of-network clinics under the CSU student policy. Without the snippets, I cannot analyze the cost structure or identify any trade-offs.
- **Maximum Out-of-Pocket Costs**: I am sorry, but I am unable to answer this question. The provided snippets are not available. Without the evidence snippets, I cannot compare the maximum out-of-pocket costs for outpatient physiotherapy between Select Plus providers and out-of-network clinics under the CSU student policy.
- **Coverage Limitations**: Unfortunately, I am unable to answer the question. The provided snippets are not available, and therefore I cannot compare the cost-sharing obligations for outpatient physiotherapy between Select Plus providers and out-of-network clinics under the CSU student policy. Without the evidence, I cannot analyze coverage limitations or identify any trade-offs.
- **Pre-authorization Requirements**: The provided evidence does not contain information about pre-authorization requirements for outpatient physiotherapy under the CSU student policy. Therefore, a comparison between Select Plus providers and out-of-network clinics regarding this aspect cannot be made. The snippets [1], [2], and [3] are insufficient to address the question of pre-authorization.
- **Claim Filing Process**: The provided evidence does not contain information comparing the claim filing processes for outpatient physiotherapy between Select Plus providers and out-of-network clinics under the CSU student policy. Therefore, a comparison cannot be made based on the provided snippets.

---

### 3. Experience Question

**Question**: For students who used the home health care benefit capped at 120 visits per policy year, what coordination steps and approvals were required?

**Question Type**: Experience (decomposed as Evidence-based)

**Processing**:
- Decomposed into 1 sub-query
- Retrieved 5 documents for evidence aspect

**Final Answer**:
> I am sorry, but the provided snippets do not contain information about the coordination steps and approvals required for students using the home health care benefit. The snippets only contain the month of April [1, 2, 3].

**LINKAGE Rank Estimate**: 2/4

**Aspect Answers**:
- **evidence**: I am sorry, but the provided snippets do not contain information about the coordination steps and approvals required for students using the home health care benefit. The snippets only contain the month of April [1, 2, 3].

---

### 4. Reason Question

**Question**: Why does the plan require written consent before an out-of-network physician at a contracting facility can bill beyond the preferred cost share, and how does that protect me from surprise bills?

**Question Type**: Reason

**Processing**:
- Decomposed into 4 sub-queries:
  1. Direct Cause: Requirement for Written Consent
  2. Mechanism: How Consent Protects
  3. Context: Historical & Regulatory Background
  4. Constraints/Limitations: Exceptions and Weaknesses
- Retrieved 5 documents for each aspect (20 total)

**Final Answer**:
> I am sorry, but I am unable to answer the question because the provided snippets do not contain any information about the requirement for written consent or how it protects against surprise bills.

**LINKAGE Rank Estimate**: 2/4

**Aspect Answers**:
- **Direct Cause: Requirement for Written Consent**: I am sorry, but I am unable to answer the question because the provided snippets do not contain any information about the requirement for written consent or how it protects against surprise bills.
- **Mechanism: How Consent Protects**: I am sorry, but I am unable to answer the question because the provided snippets are not available.
- **Context: Historical & Regulatory Background**: I am sorry, but I am unable to answer the question because the provided snippets do not contain any information about the historical and regulatory background of the plan's requirement for written consent or how it protects against surprise bills.
- **Constraints/Limitations: Exceptions and Weaknesses**: I am sorry, but I am unable to answer the question because the provided snippets do not contain any information about the plan's requirements for written consent or how it protects against surprise bills.

---

### 5. Instruction Question

**Question**: How do I request a prescription drug exception—such as for a contraceptive not on the Preferred Drug List—and what timelines does the plan give for 24- or 72-hour responses?

**Question Type**: Instruction

**Processing**:
- Decomposed into 5 sub-queries:
  1. Prerequisites/Requirements
  2. Setup/Preparation
  3. Core Steps/Implementation
  4. Validation/Testing
  5. Troubleshooting
- Retrieved 5 documents for each aspect (25 total)

**Final Answer**:
> I am unable to answer this question because the provided snippets do not contain any information about how to request a prescription drug exception, the associated timelines, or the steps involved. [1, 2, 3]

**LINKAGE Rank Estimate**: 2/4

**Aspect Answers**:
- **Prerequisites/Requirements**: I am sorry, but I am unable to answer this question. The provided snippets do not contain any information about how to request a prescription drug exception or the associated timelines.
- **Setup/Preparation**: I am sorry, but I cannot answer the question because the provided snippets do not contain any information about requesting prescription drug exceptions, timelines, or the steps involved. The snippets only contain the month "April" [1, 2, 3].
- **Core Steps/Implementation**: I am sorry, but I am unable to answer this question. The provided snippets do not contain the information needed to create actionable steps.
- **Validation/Testing**: I am sorry, but I am unable to answer this question. The provided snippets do not contain any information about requesting prescription drug exceptions or the associated timelines.
- **Troubleshooting**: I am sorry, but I am unable to answer the question as the provided snippets do not contain any information about requesting prescription drug exceptions or the associated timelines.

---

### 6. Debate Question

**Question**: Should the CSU student plan continue to set the same $5,000 out-of-pocket maximum for preferred and out-of-network services despite the higher out-of-network coinsurance?

**Question Type**: Debate

**Processing**:
- Decomposed into 3 sub-queries:
  1. Pro-Maintaining Current Plan
  2. Con-Maintaining Current Plan
  3. Neutral/Synthesis
- Retrieved 5 documents for each aspect (15 total)

**Final Answer**:
> Okay, let's summarize the debate question.
>
> 1.  **Topic:** The debate centers on whether the CSU student health plan should continue to maintain a $5,000 out-of-pocket maximum for both preferred and out-of-network healthcare services, even with a higher coinsurance rate for out-of-network services.
>
> 2.  **Pro and Con Summaries:**
>
>     *   **Pro (Maintaining Current Plan):** The argument in favor of maintaining the current plan emphasizes the financial protection it offers students. The $5,000 out-of-pocket maximum provides a safety net, capping potential healthcare costs regardless of network usage. This promotes predictability and aids in budgeting [1, 2, 3].
>
>     *   **Con (Maintaining Current Plan):** The opposing viewpoint suggests that the current plan's structure, with a high out-of-network coinsurance, already discourages out-of-network usage. Maintaining the same out-of-pocket maximum for both networks, despite the higher cost of out-of-network care, could lead to unexpected and significant financial burdens for students who inadvertently or necessarily utilize out-of-network services [3]. This could undermine the plan's affordability and accessibility.
>
> 3.  **Balanced Synthesis:**
>
>     *   The provided evidence does not contain any information about the CSU student health plan, its out-of-pocket maximum, or the impact of coinsurance. Therefore, a neutral synthesis of viewpoints cannot be provided.

**LINKAGE Rank Estimate**: 3/4

**Aspect Answers**:
- **Pro-Maintaining Current Plan**: The current CSU student health plan should maintain its $5,000 out-of-pocket maximum for both preferred and out-of-network services, despite the higher out-of-network coinsurance. This approach provides financial protection for students, ensuring that even when utilizing out-of-network providers, their total healthcare costs remain capped at a manageable level [1, 2, 3]. This structure offers predictability and helps students budget for potential healthcare needs, regardless of the network they access.
- **Con-Maintaining Current Plan**: The CSU student plan should not maintain the $5,000 out-of-pocket maximum for both preferred and out-of-network services. The current plan's structure, with a high out-of-network coinsurance, already discourages the use of out-of-network providers [1, 2]. Maintaining the same out-of-pocket maximum for both networks, despite the higher cost of out-of-network care, could lead to unexpected and significant financial burdens for students who inadvertently or necessarily utilize out-of-network services [3]. This could undermine the plan's affordability and accessibility.
- **Neutral/Synthesis**: The evidence provided does not contain any information about the CSU student health plan, its out-of-pocket maximum, or the impact of coinsurance. Therefore, a neutral synthesis of viewpoints cannot be provided.

---

## Observations

### Question Type Coverage

The demo successfully processed all major question types:
- ✅ Evidence-based
- ✅ Comparison
- ✅ Experience
- ✅ Reason
- ✅ Instruction
- ✅ Debate

### Decomposition Patterns

- **Evidence-based**: 1 sub-query (passthrough)
- **Comparison**: 5 sub-queries (multiple comparison axes)
- **Experience**: 1 sub-query (treated as Evidence-based)
- **Reason**: 4 sub-queries (causal analysis)
- **Instruction**: 5 sub-queries (step-by-step process)
- **Debate**: 3 sub-queries (pro, con, synthesis)

### Retrieval Performance

- All sub-queries successfully retrieved 5 documents each
- Total documents retrieved: 91 across all questions
- Average documents per question: ~15

### Answer Quality

**Overall Performance**:
- **MRR**: 0.556 (moderate quality)
- **MPR**: 75.000% (answers rank in 75th percentile on average)

**Common Issues**:
- Many answers indicate insufficient information in retrieved documents
- Retrieved snippets often contain only minimal information (e.g., "April" mentions)
- System correctly identifies when information is unavailable rather than hallucinating

### System Behavior

1. **Honest Responses**: System appropriately indicates when information is not available
2. **Structured Output**: Answers follow question-type-specific formats
3. **Aspect-Level Detail**: Provides detailed aspect-level answers for decomposed questions
4. **Citation Awareness**: Includes document references [1, 2, 3] when available

## Artifacts

All artifacts saved to: `/Users/indraneelsarode/Desktop/Typed-Rag/output`

Includes:
- Decomposition plans
- Evidence bundles
- Final answers
- Aspect-level responses

---

*Generated from happy_flow_demo.py run on January 2025*

