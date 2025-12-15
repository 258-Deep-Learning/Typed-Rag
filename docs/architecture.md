# Typed-RAG System Architecture

This document provides a comprehensive overview of the Typed-RAG system architecture, showing how components interact to process non-factoid questions with type-aware decomposition and retrieval.

## Table of Contents
- [High-Level System Flow](#high-level-system-flow)
- [Detailed Pipeline Architecture](#detailed-pipeline-architecture)
- [Component Interactions](#component-interactions)
- [Ablation Study Variants](#ablation-study-variants)
- [Data Flow Diagram](#data-flow-diagram)

---

## High-Level System Flow

The Typed-RAG system processes questions through five main stages: Classification, Decomposition, Retrieval, Generation, and Aggregation.

```mermaid
graph TB
    A[User Question] --> B[Question Classifier]
    B --> C{Question Type}
    C -->|Evidence| D[Query Decomposer]
    C -->|Comparison| D
    C -->|Reason| D
    C -->|Instruction| D
    C -->|Experience| D
    C -->|Debate| D
    D --> E[Sub-Queries<br/>Per Aspect]
    E --> F[Retrieval Orchestrator]
    F --> G[Evidence Bundle<br/>Per Aspect]
    G --> H[Aspect Generator]
    H --> I[Aspect Answers]
    I --> J[Answer Aggregator]
    J --> K[Final Answer]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style D fill:#f3e5f5
    style F fill:#e8f5e9
    style H fill:#fce4ec
    style J fill:#fff9c4
    style K fill:#c5e1a5
```

---

## Detailed Pipeline Architecture

This diagram shows the internal components and their interactions within each major stage.

```mermaid
graph TB
    subgraph Input
        Q[User Question]
    end
    
    subgraph "Stage 1: Classification"
        C1[Pattern Matcher]
        C2[LLM Fallback<br/>Gemini]
        C1 -->|No Match| C2
        C1 -->|Match| QT[Question Type]
        C2 --> QT
    end
    
    subgraph "Stage 2: Decomposition"
        QT --> D1{Type-Aware<br/>Strategy}
        D1 -->|Evidence| S1[Passthrough<br/>1 query]
        D1 -->|Comparison| S2[Subjects + Axes<br/>3-5 queries]
        D1 -->|Reason| S3[Causal Factors<br/>2-4 queries]
        D1 -->|Instruction| S4[Sequential Steps<br/>3-5 queries]
        D1 -->|Experience| S5[Topical Angles<br/>2-4 queries]
        D1 -->|Debate| S6[Opposing Views<br/>2-3 queries]
        S1 --> SQ[Sub-Queries]
        S2 --> SQ
        S3 --> SQ
        S4 --> SQ
        S5 --> SQ
        S6 --> SQ
    end
    
    subgraph "Stage 3: Retrieval"
        SQ --> R1[BGE Embedder]
        R1 --> R2[Dense Retrieval<br/>FAISS/Pinecone]
        R2 --> R3[Top-K Docs<br/>k=50]
        R3 --> R4{Reranking?}
        R4 -->|Yes| R5[Cross-Encoder<br/>Reranker]
        R4 -->|No| R6[Final Top-K<br/>k=5]
        R5 --> R6
        R6 --> EB[Evidence Bundle<br/>Organized by Aspect]
    end
    
    subgraph "Stage 4: Generation"
        EB --> G1[Aspect Generator<br/>Per Sub-Query]
        G1 --> G2[LLM Generation<br/>Gemini]
        G2 --> AA[Aspect Answers]
    end
    
    subgraph "Stage 5: Aggregation"
        AA --> A1[Answer Aggregator]
        A1 --> A2[Type-Aware<br/>Formatting]
        A2 --> FA[Final Answer]
    end
    
    Q --> C1
    
    style Q fill:#e3f2fd
    style QT fill:#fff3e0
    style SQ fill:#f3e5f5
    style EB fill:#e8f5e9
    style AA fill:#fce4ec
    style FA fill:#c5e1a5
```

---

## Component Interactions

### Question Classifier

```mermaid
graph LR
    Q[Question] --> P[Pattern<br/>Matcher]
    P -->|Match Found| T1[Type: Evidence]
    P -->|Match Found| T2[Type: Comparison]
    P -->|Match Found| T3[Type: Reason]
    P -->|Match Found| T4[Type: Instruction]
    P -->|Match Found| T5[Type: Experience]
    P -->|Match Found| T6[Type: Debate]
    P -->|No Match| L[LLM Classifier<br/>Gemini]
    L --> T1
    L --> T2
    L --> T3
    L --> T4
    L --> T5
    L --> T6
    
    style P fill:#fff3e0
    style L fill:#ffccbc
```

**Pattern Examples:**
- Evidence: "What is...", "Define...", "Explain..."
- Comparison: "vs", "compared to", "difference between"
- Reason: "Why...", "What causes...", "How come..."
- Instruction: "How to...", "Steps to...", "Guide for..."
- Experience: "experiences with", "opinions on", "reviews of"
- Debate: "pros and cons", "advantages and disadvantages"

---

### Retrieval Orchestrator

```mermaid
sequenceDiagram
    participant DP as Decomposition Plan
    participant RO as Retrieval Orchestrator
    participant EMB as BGE Embedder
    participant VS as Vector Store
    participant RE as Reranker
    participant EB as Evidence Bundle
    
    DP->>RO: Sub-queries per aspect
    loop For each sub-query
        RO->>EMB: Embed sub-query
        EMB-->>RO: Query embedding
        RO->>VS: Search (top_k=50)
        VS-->>RO: Retrieved documents
        alt Reranking enabled
            RO->>RE: Rerank documents
            RE-->>RO: Reranked top-5
        else No reranking
            RO->>RO: Select top-5
        end
        RO->>EB: Store aspect evidence
    end
    EB-->>RO: Complete evidence bundle
```

---

## Ablation Study Variants

This diagram shows how different ablation variants modify the pipeline by disabling specific components.

```mermaid
graph TB
    subgraph "Full Typed-RAG (Baseline)"
        F1[Classification ✓] --> F2[Decomposition ✓]
        F2 --> F3[Retrieval ✓]
        F3 --> F4[Generation ✓]
    end
    
    subgraph "No Classification"
        N1[Classification ✗<br/>Force Evidence-based] --> N2[Decomposition ✓]
        N2 --> N3[Retrieval ✓]
        N3 --> N4[Generation ✓]
    end
    
    subgraph "No Decomposition"
        D1[Classification ✓] --> D2[Decomposition ✗<br/>Single Query]
        D2 --> D3[Retrieval ✓]
        D3 --> D4[Generation ✓]
    end
    
    subgraph "No Retrieval"
        R1[Classification ✓] --> R2[Decomposition ✓]
        R2 --> R3[Retrieval ✗<br/>Pure LLM]
        R3 --> R4[Generation ✓]
    end
    
    style F1 fill:#c8e6c9
    style F2 fill:#c8e6c9
    style F3 fill:#c8e6c9
    style F4 fill:#c8e6c9
    
    style N1 fill:#ffcdd2
    style D2 fill:#ffcdd2
    style R3 fill:#ffcdd2
```

**Component Impact:**
- **Classification**: ~1s overhead, enables type-aware decomposition
- **Decomposition**: Critical for multi-aspect questions, improves coverage
- **Retrieval**: 48% speedup vs pure LLM (3.81s vs 5.62s), provides evidence

---

## Data Flow Diagram

### Question Processing Flow

```mermaid
flowchart TD
    Start([User Input]) --> Check{Data Available?}
    Check -->|No| Setup[Setup Vector Store]
    Check -->|Yes| Classify
    Setup --> Classify[Classify Question Type]
    
    Classify --> Decompose[Decompose into<br/>Sub-Queries]
    Decompose --> Cache1{Cached?}
    Cache1 -->|Yes| LoadCache1[Load Cached Plan]
    Cache1 -->|No| Generate1[Generate Plan]
    Generate1 --> SaveCache1[Save to Cache]
    LoadCache1 --> Retrieve
    SaveCache1 --> Retrieve
    
    Retrieve[Retrieve Evidence<br/>Per Aspect] --> Cache2{Cached?}
    Cache2 -->|Yes| LoadCache2[Load Cached Evidence]
    Cache2 -->|No| Execute[Execute Retrieval]
    Execute --> SaveCache2[Save to Cache]
    LoadCache2 --> GenerateAns
    SaveCache2 --> GenerateAns
    
    GenerateAns[Generate Aspect<br/>Answers] --> Cache3{Cached?}
    Cache3 -->|Yes| LoadCache3[Load Cached Answers]
    Cache3 -->|No| LLMGen[LLM Generation]
    LLMGen --> SaveCache3[Save to Cache]
    LoadCache3 --> Aggregate
    SaveCache3 --> Aggregate
    
    Aggregate[Aggregate into<br/>Final Answer] --> Output([Final Answer])
    
    style Start fill:#e3f2fd
    style Classify fill:#fff3e0
    style Decompose fill:#f3e5f5
    style Retrieve fill:#e8f5e9
    style GenerateAns fill:#fce4ec
    style Aggregate fill:#fff9c4
    style Output fill:#c5e1a5
```

---

## Vector Store Architecture

```mermaid
graph TB
    subgraph "Document Ingestion"
        D1[Raw Documents] --> D2[Document Chunker]
        D2 --> D3[Chunks<br/>~500 tokens]
        D3 --> D4[BGE Embedder]
        D4 --> D5[768-dim Vectors]
    end
    
    subgraph "Index Building"
        D5 --> I1{Backend}
        I1 -->|FAISS| I2[FAISS Index<br/>Local]
        I1 -->|Pinecone| I3[Pinecone Index<br/>Cloud]
    end
    
    subgraph "Query Processing"
        Q1[Query] --> Q2[BGE Embedder]
        Q2 --> Q3[Query Vector]
        Q3 --> I2
        Q3 --> I3
        I2 --> R1[Top-K Results]
        I3 --> R1
    end
    
    style D1 fill:#e3f2fd
    style D5 fill:#fff3e0
    style I2 fill:#e8f5e9
    style I3 fill:#e8f5e9
    style R1 fill:#c5e1a5
```

---

## Evaluation Pipeline

```mermaid
graph LR
    subgraph "Input"
        Q[Questions<br/>dev6.jsonl]
    end
    
    subgraph "Systems"
        S1[LLM-Only]
        S2[RAG Baseline]
        S3[Typed-RAG]
        S4[Ablation Variants]
    end
    
    subgraph "Evaluation"
        E1[LINKAGE<br/>MRR/MPR]
        E2[Classifier<br/>Accuracy/F1]
        E3[Latency<br/>Measurement]
        E4[Success Rate]
    end
    
    subgraph "Output"
        O1[Results JSON]
        O2[Markdown Reports]
        O3[Plots/Charts]
    end
    
    Q --> S1
    Q --> S2
    Q --> S3
    Q --> S4
    
    S1 --> E1
    S2 --> E1
    S3 --> E1
    S4 --> E1
    
    S3 --> E2
    
    S1 --> E3
    S2 --> E3
    S3 --> E3
    S4 --> E3
    
    S1 --> E4
    S2 --> E4
    S3 --> E4
    S4 --> E4
    
    E1 --> O1
    E2 --> O1
    E3 --> O1
    E4 --> O1
    
    O1 --> O2
    O1 --> O3
    
    style Q fill:#e3f2fd
    style S3 fill:#c8e6c9
    style E1 fill:#fff3e0
    style O3 fill:#c5e1a5
```

---

## Technology Stack

```mermaid
graph TB
    subgraph "LLM & Embedding"
        L1[Google Gemini 2.0<br/>Flash Lite]
        L2[BGE-small-en-v1.5<br/>Embeddings]
    end
    
    subgraph "Vector Stores"
        V1[FAISS<br/>Local Index]
        V2[Pinecone<br/>Cloud Index]
    end
    
    subgraph "Frameworks"
        F1[LangChain<br/>Orchestration]
        F2[Transformers<br/>Models]
        F3[Sentence-Transformers<br/>Reranking]
    end
    
    subgraph "UI & Evaluation"
        U1[Streamlit<br/>Demo UI]
        U2[Matplotlib/Seaborn<br/>Plots]
        U3[Pytest<br/>Testing]
    end
    
    style L1 fill:#e3f2fd
    style L2 fill:#e3f2fd
    style V1 fill:#e8f5e9
    style V2 fill:#e8f5e9
    style F1 fill:#fff3e0
    style U1 fill:#f3e5f5
```

---

## Key Design Decisions

### 1. Hybrid Classification
- **Pattern Matching (60%)**: Fast, deterministic, zero-cost
- **LLM Fallback (40%)**: Handles edge cases, high accuracy
- **Rationale**: Balance speed, cost, and accuracy

### 2. Type-Aware Decomposition
- **6 Question Types**: Evidence, Comparison, Reason, Instruction, Experience, Debate
- **Custom Strategies**: Each type has tailored decomposition logic
- **Rationale**: Better coverage of multi-aspect questions

### 3. Dense + Reranking Retrieval
- **Stage 1**: Dense retrieval (BGE + FAISS/Pinecone) - Broad recall
- **Stage 2**: Cross-encoder reranking - Precision refinement
- **Rationale**: Balance recall and precision

### 4. Aspect-Level Generation
- **Per-Aspect Answers**: Generate focused answers for each sub-query
- **Final Aggregation**: Combine with type-aware formatting
- **Rationale**: Better structured, comprehensive responses

---

## Performance Characteristics

| Component | Latency | Accuracy | Offline | Cost |
|-----------|---------|----------|---------|------|
| **Classification** | 200-500ms | 90-95% | Hybrid | $0.001/query |
| **Decomposition** | 100-300ms | N/A | ✓ | Free |
| **Retrieval** | 100-200ms | MRR: 0.47 | ✓ | Free |
| **Reranking** | 50-100ms | +10-15% | ✓ | Free |
| **Generation** | 500-1000ms | N/A | ✗ | $0.002/query |
| **Total** | ~2-4s | N/A | Hybrid | ~$0.003/query |

---

## Caching Strategy

```mermaid
graph LR
    Q[Query Hash] --> C1{Decomposition<br/>Cache?}
    C1 -->|Hit| L1[Load Plan]
    C1 -->|Miss| G1[Generate Plan]
    G1 --> S1[Save Plan]
    
    L1 --> C2{Evidence<br/>Cache?}
    S1 --> C2
    C2 -->|Hit| L2[Load Evidence]
    C2 -->|Miss| G2[Retrieve Evidence]
    G2 --> S2[Save Evidence]
    
    L2 --> C3{Answer<br/>Cache?}
    S2 --> C3
    C3 -->|Hit| L3[Load Answer]
    C3 -->|Miss| G3[Generate Answer]
    G3 --> S3[Save Answer]
    
    L3 --> FA[Final Answer]
    S3 --> FA
    
    style C1 fill:#fff3e0
    style C2 fill:#fff3e0
    style C3 fill:#fff3e0
    style FA fill:#c5e1a5
```

**Cache Locations:**
- Decomposition: `cache/decomposition/{hash}.json`
- Evidence: `cache/evidence/{hash}.json`
- Answers: `cache/answers/{hash}.json`
- Final: `cache/final_answers/{hash}.json`

---

## References

- **Paper**: "Typed-RAG: Type-Aware Decomposition of Non-Factoid Questions"
- **Code**: [GitHub Repository](https://github.com/yourusername/Typed-Rag)
- **Demo**: Run `streamlit run app.py`
- **Evaluation**: See [EVALUATION.md](../EVALUATION.md)
