#!/usr/bin/env python3
"""
Typed-RAG Streamlit Demo App

Interactive UI to demonstrate:
1. Query interface with system selection
2. Typed-RAG pipeline visualization
3. Evaluation metrics and ablation study results
4. Detailed evidence display with full document text
"""

import streamlit as st
import time
import json
import hashlib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any

# Load environment variables
load_dotenv()

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from typed_rag.rag_system import DataType, ask_typed_question
from typed_rag.classifier import classify_question
from typed_rag.decompose import decompose_question
from typed_rag.core.keys import get_fastest_model
from typed_rag.retrieval.orchestrator import EvidenceBundle
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# Page config
st.set_page_config(
    page_title="Typed-RAG Demo",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #333;
    }
    .pipeline-step {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
        color: #333;
    }
    .aspect-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        color: #333;
    }
    .doc-card {
        background-color: #fafafa;
        padding: 0.8rem;
        border-radius: 6px;
        border: 1px solid #eee;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #333;
    }
    .metric-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .evaluation-table {
        width: 100%;
        border-collapse: collapse;
    }
    .evaluation-table th, .evaluation-table td {
        padding: 0.5rem;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .evaluation-table th {
        background-color: #f0f0f0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

def generate_question_id(question: str) -> str:
    """Generate question ID from question text."""
    return hashlib.md5(question.encode()).hexdigest()[:12]


def load_evidence_bundle(question_id: str, repo_root: Optional[Path] = None) -> Optional[EvidenceBundle]:
    """Load evidence bundle from output directory."""
    if repo_root is None:
        repo_root = Path(__file__).parent
    
    evidence_path = repo_root / "output" / f"{question_id}_evidence_bundle.json"
    
    if not evidence_path.exists():
        return None
    
    try:
        with open(evidence_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return EvidenceBundle.from_dict(data)
    except Exception as e:
        st.warning(f"Could not load evidence bundle: {e}")
        return None


def load_evaluation_results(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load evaluation results from results directory."""
    if repo_root is None:
        repo_root = Path(__file__).parent
    
    results_dir = repo_root / "results"
    evaluation_data = {}
    
    # Load full linkage evaluation (all 12 systems)
    linkage_path = results_dir / "full_linkage_evaluation.json"
    if linkage_path.exists():
        try:
            with open(linkage_path, 'r') as f:
                evaluation_data["linkage"] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load linkage evaluation: {e}")
    
    # Load classifier evaluation
    classifier_path = results_dir / "classifier_evaluation.json"
    if classifier_path.exists():
        try:
            with open(classifier_path, 'r') as f:
                evaluation_data["classifier"] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load classifier evaluation: {e}")
    
    # Load other evaluation files
    for eval_file in ["decomposition_evaluation.json", "generation_evaluation.json", "retrieval_evaluation.json"]:
        eval_path = results_dir / eval_file
        if eval_path.exists():
            try:
                with open(eval_path, 'r') as f:
                    key = eval_file.replace("_evaluation.json", "")
                    evaluation_data[key] = json.load(f)
            except Exception as e:
                pass
    
    return evaluation_data


def load_ablation_results(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load ablation study results."""
    if repo_root is None:
        repo_root = Path(__file__).parent
    
    ablation_data = {}
    
    # Load summary
    summary_path = repo_root / "results" / "ablation" / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                ablation_data["summary"] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load ablation summary: {e}")
    
    # Load linkage evaluation for ablation
    ablation_linkage_path = repo_root / "results" / "ablation_linkage_evaluation.json"
    if ablation_linkage_path.exists():
        try:
            with open(ablation_linkage_path, 'r') as f:
                ablation_data["linkage"] = json.load(f)
        except Exception as e:
            st.warning(f"Could not load ablation linkage evaluation: {e}")
    
    return ablation_data


def format_document(doc: Dict[str, Any], include_text: bool = True, max_text_length: int = 300) -> str:
    """Format document for display."""
    title = doc.get("title", "Untitled")
    url = doc.get("url", "")
    score = doc.get("score", 0.0)
    text = doc.get("text", "")
    
    parts = [f"**{title}**"]
    if score > 0:
        parts.append(f"Score: {score:.3f}")
    if url:
        parts.append(f"Source: {url}")
    if include_text and text:
        text_preview = text[:max_text_length] + "..." if len(text) > max_text_length else text
        parts.append(f"\n{text_preview}")
    
    return "\n".join(parts)


def generate_llm_only_answer(question: str, model_name: str) -> tuple[str, float]:
    """Generate answer using only LLM without retrieval."""
    start = time.time()
    
    # Check if it's a Groq model
    if model_name.startswith("groq/"):
        from langchain_groq import ChatGroq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "Error: GROQ_API_KEY not set", 0.0
        
        actual_model = model_name.replace("groq/", "")
        llm = ChatGroq(
            model=actual_model,
            groq_api_key=groq_api_key,
            temperature=0.0
        )
    else:
        # Google Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            return "Error: GOOGLE_API_KEY not set", 0.0
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.0
        )
    
    prompt = f"""Answer the following question concisely and accurately.
Do not make up information. If you're unsure, say so.

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    answer = response.content.strip()
    
    elapsed = time.time() - start
    return answer, elapsed


def generate_rag_baseline_answer(question: str, model_name: str, data_type: DataType) -> tuple[str, float, list]:
    """Generate answer using simple RAG (retrieve then generate)."""
    start = time.time()
    
    # For now, use typed-rag with no decomposition (simplified)
    # In production, this would use a simpler retrieval pipeline
    result = ask_typed_question(
        question,
        data_type,
        model_name=model_name,
        rerank=False,
        use_llm=True,
        save_artifacts=False
    )
    
    elapsed = time.time() - start
    
    # Extract retrieved docs (FIXED: use 'sources' instead of 'evidence')
    docs = []
    if hasattr(result, 'aspects') and result.aspects:
        for aspect in result.aspects[:1]:  # Just first aspect for baseline
            if 'sources' in aspect:  # Changed from 'evidence'
                docs.extend(aspect['sources'][:5])
    
    return result.answer, elapsed, docs


def main():
    # Header
    st.markdown('<div class="main-header">🔍 Typed-RAG Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Aspect Question Answering with Type-Aware Decomposition</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query Interface", "📊 System Evaluation", "🧩 Component Quality", "🔬 Ablation Study"])
    
    # Tab 1: Query Interface
    with tab1:
        _render_query_interface()
    
    # Tab 2: System Evaluation (LINKAGE metrics)
    with tab2:
        _render_system_evaluation()
    
    # Tab 3: Component Quality Evaluation
    with tab3:
        _render_component_evaluation()
    
    # Tab 4: Ablation Study
    with tab4:
        _render_ablation_study()


def _render_query_interface():
    """Render the main query interface."""
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    # =========================================================================
    # SECTION 1: Interactive Query Interface (Left Sidebar)
    # =========================================================================
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., Why did the campus police establish in 1958?",
            help="Ask any question - the system will classify its type and decompose it"
        )
        
        # System selection
        system = st.selectbox(
            "Select System:",
            ["LLM-Only", "RAG Baseline", "Typed-RAG"],
            index=2,
            help="Choose which system to use for answering"
        )
        
        # Model selection
        model = st.selectbox(
            "Select Model:",
            [
                "gemini-2.5-flash",
                "gemini-2.0-flash-lite",
                "groq/llama-3.3-70b-versatile",
                "groq/llama-3.1-8b-instant"
            ],
            help="Language model to use for generation"
        )
        
        # Data source
        source = st.radio(
            "Data Source:",
            ["Wikipedia", "Own Documents"],
            index=0,
            help="Choose the knowledge base to retrieve from"
        )
        
        # Example questions
        st.markdown("#### 💡 Example Questions")
        examples = {
            "Evidence": "What is quantum computing?",
            "Comparison": "How do Python and Java differ?",
            "Reason": "Why was the campus police established?",
            "Instruction": "How were humanists able to identify development?",
            "Experience": "What is unique about Tbilisi?",
        }
        
        selected_example = st.selectbox(
            "Or try an example:",
            [""] + list(examples.values()),
            format_func=lambda x: "Select an example..." if x == "" else x
        )
        
        if selected_example and selected_example != "":
            question = selected_example
        
        # Submit button
        submit = st.button("🚀 Ask Question", type="primary")
    
    # =========================================================================
    # SECTION 2: Results and Pipeline Visualization (Right Panel)
    # =========================================================================
    
    with col2:
        if submit and question:
            # Setup data type
            source_key = "wikipedia" if source == "Wikipedia" else "own_docs"
            data_type = DataType("faiss", source_key)
            
            # Process based on system selection
            if system == "LLM-Only":
                st.markdown("### 💬 Answer")
                
                with st.spinner("Generating answer..."):
                    answer, elapsed = generate_llm_only_answer(question, model)
                
                # Display answer
                st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{answer}</div>', unsafe_allow_html=True)
                st.info(f"⏱️ Time taken: {elapsed:.2f}s")
                
                st.warning("⚠️ LLM-Only: No retrieval - answer based solely on model's training data")
            
            elif system == "RAG Baseline":
                st.markdown("### 💬 Answer")
                
                with st.spinner("Retrieving documents and generating answer..."):
                    answer, elapsed, docs = generate_rag_baseline_answer(question, model, data_type)
                
                # Display answer
                st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{answer}</div>', unsafe_allow_html=True)
                st.info(f"⏱️ Time taken: {elapsed:.2f}s")
                
                # Show retrieved docs
                with st.expander(f"📄 Retrieved Documents ({len(docs)} docs)", expanded=False):
                    for i, doc in enumerate(docs[:5], 1):
                        title = doc.get("title", "Untitled")
                        url = doc.get("url", "")
                        score = doc.get("score", 0.0)
                        doc_info = f'<div class="doc-card"><strong>Doc {i}:</strong> {title}'
                        if score > 0:
                            doc_info += f' <small>(Score: {score:.3f})</small>'
                        if url:
                            doc_info += f'<br><small>Source: {url}</small>'
                        doc_info += '</div>'
                        st.markdown(doc_info, unsafe_allow_html=True)
            
            elif system == "Typed-RAG":
                st.markdown("### 🔬 Typed-RAG Pipeline")
                
                # Step 1: Classification
                with st.expander("📋 Step 1: Question Classification", expanded=True):
                    with st.spinner("Classifying question type..."):
                        q_type = classify_question(question, use_llm=True)
                    
                    st.markdown(f'<div class="pipeline-step"><strong>Question:</strong> {question}<br><strong>Classified Type:</strong> <span style="color: #1f77b4; font-weight: bold;">{q_type}</span></div>', unsafe_allow_html=True)
                    
                    type_descriptions = {
                        "Evidence-based": "Factual question requiring evidence-based answer",
                        "Comparison": "Comparing multiple entities or concepts",
                        "Reason": "Asking for causes, reasons, or explanations",
                        "Instruction": "How-to or procedural question",
                        "Experience": "Subjective or experiential question",
                        "Debate": "Opinion or argumentative question"
                    }
                    st.info(f"ℹ️ {type_descriptions.get(q_type, 'Unknown type')}")
                
                # Step 2: Decomposition
                with st.expander("🔀 Step 2: Multi-Aspect Decomposition", expanded=True):
                    with st.spinner("Decomposing into sub-questions..."):
                        decomposition = decompose_question(question, q_type)
                        # DecompositionPlan has sub_queries, not aspects
                        sub_queries = decomposition.sub_queries if hasattr(decomposition, 'sub_queries') else []
                    
                    st.success(f"✓ Decomposed into {len(sub_queries)} aspects")
                    
                    for i, sq in enumerate(sub_queries, 1):
                        aspect_name = sq.aspect if hasattr(sq, 'aspect') else sq.get("aspect", "Unknown") if isinstance(sq, dict) else "Unknown"
                        sub_query = sq.query if hasattr(sq, 'query') else sq.get("query", "") if isinstance(sq, dict) else ""
                        st.markdown(f'<div class="aspect-card"><strong>Aspect {i}:</strong> {aspect_name}<br><em>{sub_query}</em></div>', unsafe_allow_html=True)
                
                # Step 3: Retrieval + Generation
                with st.expander("🔍 Step 3: Retrieval & Generation", expanded=True):
                    with st.spinner("Retrieving evidence and generating answers..."):
                        start = time.time()
                        try:
                            result = ask_typed_question(
                                question,
                                data_type,
                                model_name=model,
                                rerank=False,
                                use_llm=True,
                                save_artifacts=True  # Enable to load evidence bundles
                            )
                            elapsed = time.time() - start
                        except Exception as e:
                            if "429" in str(e) or "quota" in str(e).lower():
                                st.error("⚠️ **API Quota Exceeded**: You've reached the daily limit of 200 requests for Gemini API. Please try again tomorrow or use a different API key.")
                                st.stop()
                            else:
                                st.error(f"❌ Error: {str(e)}")
                                st.stop()
                    
                    # Load evidence bundle for full document text
                    evidence_bundle = None
                    if hasattr(result, 'question_id'):
                        evidence_bundle = load_evidence_bundle(result.question_id)
                    
                    if result.aspects:
                        st.success(f"✓ Retrieved and generated answers for {len(result.aspects)} aspects")
                        
                        for i, aspect in enumerate(result.aspects, 1):
                            with st.container():
                                aspect_name = aspect.get('aspect', 'Unknown')
                                st.markdown(f"**Aspect {i}: {aspect_name}**")
                                
                                # Show retrieved docs for this aspect
                                # Try to get full text from evidence bundle, fallback to sources metadata
                                aspect_docs = []
                                if evidence_bundle:
                                    # Find matching aspect in evidence bundle
                                    for ev in evidence_bundle.evidence:
                                        if ev.aspect == aspect_name:
                                            aspect_docs = [doc.to_dict() for doc in ev.documents]
                                            break
                                
                                # Fallback to sources if evidence bundle not available
                                if not aspect_docs and 'sources' in aspect:
                                    aspect_docs = aspect['sources']
                                
                                if aspect_docs:
                                    num_docs = len(aspect_docs)
                                    with st.expander(f"📄 {num_docs} documents retrieved", expanded=False):
                                        for j, doc in enumerate(aspect_docs[:5], 1):
                                            doc_text = format_document(doc, include_text=True, max_text_length=200)
                                            st.markdown(f'<div class="doc-card"><small>{doc_text}</small></div>', unsafe_allow_html=True)
                                
                                # Show aspect answer
                                if 'answer' in aspect:
                                    st.markdown(f"*Answer:* {aspect['answer']}")
                                
                                st.divider()
                
                # Step 4: Final Answer
                st.markdown("### 💬 Final Aggregated Answer")
                st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)
                st.info(f"⏱️ Total time: {elapsed:.2f}s")
                
                # Summary stats
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Question Type", q_type)
                with col_b:
                    st.metric("Aspects", len(result.aspects) if result.aspects else 0)
                with col_c:
                    # FIXED: count 'sources' instead of 'evidence'
                    total_docs = sum(len(a.get('sources', [])) for a in result.aspects) if result.aspects else 0
                    st.metric("Total Docs", total_docs)
        
        elif submit and not question:
            st.warning("⚠️ Please enter a question")
        
        else:
            # Welcome message
            st.markdown("### 👋 Welcome to Typed-RAG!")
            st.markdown("""
            This demo showcases a **type-aware multi-aspect RAG system** that:
            
            1. **Classifies** questions into 6 types (Evidence, Comparison, Reason, etc.)
            2. **Decomposes** questions into focused sub-questions per aspect
            3. **Retrieves** relevant evidence for each aspect separately
            4. **Generates** aspect-level answers
            5. **Aggregates** into a comprehensive final answer
            
            **Try it out:**
            - Enter your question in the left panel
            - Select a system (LLM-Only, RAG Baseline, or Typed-RAG)
            - Click "Ask Question" to see the magic! ✨
            """)
            
            st.info("💡 **Tip:** Try different question types to see how Typed-RAG adapts its decomposition strategy!")


def _render_system_evaluation():
    """Render comprehensive system evaluation with LINKAGE metrics for all 12 systems."""
    st.markdown("### 📊 System Evaluation (LINKAGE Metrics)")
    st.markdown("""Comprehensive evaluation of all systems using **LINKAGE** framework:  
    - **MRR** (Mean Reciprocal Rank): Position-aware accuracy metric  
    - **MPR** (Mean Precision Rate): Percentage of questions with relevant answers  
    """)
    
    repo_root = Path(__file__).parent
    eval_data = load_evaluation_results(repo_root)
    
    if not eval_data or "linkage" not in eval_data:
        st.info("No evaluation results found. Run: `python scripts/evaluate_linkage.py`")
        return
    
    linkage = eval_data["linkage"]
    
    # Organize systems by model
    model_groups = {
        "Gemini 2.5 Flash": ["llm_only_gemini_dev100", "rag_baseline_gemini_dev100", "typed_rag_gemini_dev100"],
        "Gemini 2.0 Flash-Lite": ["llm_only_gemini2lite_dev100", "rag_baseline_gemini2lite_dev100", "typed_rag_gemini2lite_dev100"],
        "Llama 3.3 70B": ["llm_only_llama_dev100", "rag_baseline_llama_dev100", "typed_rag_llama_dev100"],
        "Llama 3.1 8B": ["llm_only_llama31_8b_dev100", "rag_baseline_llama31_8b_dev100", "typed_rag_llama31_8b_dev100"]
    }
    
    # Overall comparison table
    st.markdown("#### 🏆 All Systems Comparison")
    all_metrics = []
    for model_name, systems in model_groups.items():
        for system_key in systems:
            if system_key in linkage and "overall" in linkage[system_key]:
                overall = linkage[system_key]["overall"]
                system_type = "LLM-Only" if "llm_only" in system_key else "RAG Baseline" if "rag_baseline" in system_key else "Typed-RAG"
                all_metrics.append({
                    "Model": model_name,
                    "System": system_type,
                    "MRR": overall.get("mrr", 0.0),
                    "MPR (%)": overall.get("mpr", 0.0),
                    "Questions": overall.get("questions", 0)
                })
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        st.dataframe(df.style.format({"MRR": "{:.4f}", "MPR (%)": "{:.2f}"}), use_container_width=True)
    
    # Per-model breakdown
    st.markdown("---")
    st.markdown("#### 📈 Per-Model Performance")
    
    for model_name, systems in model_groups.items():
        with st.expander(f"🔸 {model_name}", expanded=False):
            model_metrics = []
            for system_key in systems:
                if system_key in linkage and "overall" in linkage[system_key]:
                    overall = linkage[system_key]["overall"]
                    system_type = "LLM-Only" if "llm_only" in system_key else "RAG Baseline" if "rag_baseline" in system_key else "Typed-RAG"
                    model_metrics.append({
                        "System": system_type,
                        "MRR": overall.get("mrr", 0.0),
                        "MPR (%)": overall.get("mpr", 0.0)
                    })
            
            if model_metrics:
                cols = st.columns(3)
                for i, metric in enumerate(model_metrics):
                    with cols[i]:
                        st.metric(
                            metric["System"],
                            f"MRR: {metric['MRR']:.4f}",
                            f"MPR: {metric['MPR (%)']:.2f}%"
                        )
                
                # Per-category breakdown
                st.markdown("**By Question Category:**")
                category_data = []
                for system_key in systems:
                    if system_key in linkage and "by_category" in linkage[system_key]:
                        system_type = "LLM-Only" if "llm_only" in system_key else "RAG Baseline" if "rag_baseline" in system_key else "Typed-RAG"
                        by_cat = linkage[system_key]["by_category"]
                        for cat_name, cat_data in by_cat.items():
                            category_data.append({
                                "System": system_type,
                                "Category": cat_name,
                                "MRR": cat_data.get("mrr", 0.0),
                                "MPR (%)": cat_data.get("mpr", 0.0),
                                "Count": cat_data.get("count", 0)
                            })
                
                if category_data:
                    cat_df = pd.DataFrame(category_data)
                    st.dataframe(cat_df.style.format({"MRR": "{:.4f}", "MPR (%)": "{:.2f}"}), use_container_width=True)


def _render_component_evaluation():
    """Render component-wise quality evaluation."""
    st.markdown("### 🧩 Component Quality Evaluation")
    st.markdown("Detailed analysis of individual Typed-RAG components: Classification, Decomposition, and Retrieval.")
    
    repo_root = Path(__file__).parent
    eval_data = load_evaluation_results(repo_root)
    
    if not eval_data:
        st.info("No evaluation results found. Run evaluation scripts to generate results.")
        return
    
    # Display classifier evaluation
    if "classifier" in eval_data:
        st.markdown("#### 🏷️ Question Classification")
        st.markdown("Pattern-based classifier for question type detection (Evidence, Reason, Comparison, etc.)")
        classifier = eval_data["classifier"]
        
        if "pattern_only" in classifier:
            pattern_data = classifier["pattern_only"]
            if "accuracy" in pattern_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Accuracy", f"{pattern_data['accuracy']:.1%}")
                with col2:
                    if "macro_f1" in pattern_data:
                        st.metric("Macro F1", f"{pattern_data['macro_f1']:.1%}")
                with col3:
                    if "weighted_f1" in pattern_data:
                        st.metric("Weighted F1", f"{pattern_data['weighted_f1']:.1%}")
            
            if "per_type" in pattern_data:
                st.markdown("**Per-Type Performance:**")
                type_data = pattern_data["per_type"]
                type_df = pd.DataFrame([
                    {
                        "Type": qtype,
                        "Precision": data.get("precision", 0.0),
                        "Recall": data.get("recall", 0.0),
                        "F1": data.get("f1", 0.0),
                        "Support": data.get("support", 0)
                    }
                    for qtype, data in type_data.items()
                ])
                st.dataframe(type_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}), use_container_width=True)
    else:
        st.info("⚠️ Classification evaluation not found. Run: `python scripts/evaluate_classifier.py`")
    
    st.markdown("---")
    
    # Display decomposition evaluation
    if "decomposition" in eval_data:
        st.markdown("#### 🔀 Question Decomposition")
        st.markdown("Quality of breaking down complex questions into sub-queries.")
        decomp = eval_data["decomposition"]
        
        if "overall" in decomp:
            overall = decomp["overall"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Quality", f"{overall.get('avg_quality', 0.0):.2f}/5.0")
            with col2:
                st.metric("Questions Evaluated", overall.get("count", 0))
            with col3:
                st.metric("Excellent (5/5)", f"{overall.get('excellent_pct', 0.0):.1f}%")
        
        if "by_category" in decomp:
            st.markdown("**By Question Category:**")
            cat_data = decomp["by_category"]
            cat_df = pd.DataFrame([
                {
                    "Category": cat,
                    "Avg Quality": data.get("avg_quality", 0.0),
                    "Count": data.get("count", 0)
                }
                for cat, data in cat_data.items()
            ])
            st.dataframe(cat_df.style.format({"Avg Quality": "{:.2f}"}), use_container_width=True)
    else:
        st.info("⚠️ Decomposition evaluation not found. Run: `python scripts/evaluate_decomposition.py`")
    
    st.markdown("---")
    
    # Display retrieval evaluation
    if "retrieval" in eval_data:
        st.markdown("#### 🔍 Document Retrieval")
        st.markdown("Quality of retrieved evidence documents for answering questions.")
        retrieval = eval_data["retrieval"]
        
        if "overall" in retrieval:
            overall = retrieval["overall"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Quality", f"{overall.get('avg_quality', 0.0):.2f}/5.0")
            with col2:
                st.metric("Queries Evaluated", overall.get("count", 0))
            with col3:
                st.metric("High Quality (4-5)", f"{overall.get('high_quality_pct', 0.0):.1f}%")
        
        if "by_category" in retrieval:
            st.markdown("**By Question Category:**")
            cat_data = retrieval["by_category"]
            cat_df = pd.DataFrame([
                {
                    "Category": cat,
                    "Avg Quality": data.get("avg_quality", 0.0),
                    "Count": data.get("count", 0)
                }
                for cat, data in cat_data.items()
            ])
            st.dataframe(cat_df.style.format({"Avg Quality": "{:.2f}"}), use_container_width=True)
    else:
        st.info("⚠️ Retrieval evaluation not found. Run: `python scripts/evaluate_retrieval.py`")


def _render_ablation_study():
    """Render ablation study visualization."""
    st.markdown("### 🔬 Ablation Study Results")
    st.markdown("Comparison of Typed-RAG variants to evaluate component impact.")
    
    repo_root = Path(__file__).parent
    ablation_data = load_ablation_results(repo_root)
    
    if not ablation_data:
        st.info("No ablation study results found. Run ablation study script to generate results.")
        return
    
    # Display linkage evaluation for ablation (use this as primary data source)
    if "linkage" in ablation_data:
        st.markdown("#### Performance Comparison")
        st.markdown("""Ablation study evaluates the impact of removing key components.  
        **Lower MRR/MPR** when a component is removed indicates that component is important.""")
        
        linkage = ablation_data["linkage"]
        
        ablation_metrics = []
        for variant_name, variant_data in linkage.items():
            if isinstance(variant_data, dict) and "overall" in variant_data:
                overall = variant_data["overall"]
                # Map variant names to more descriptive labels
                label_map = {
                    "full": "✅ Full System (Baseline)",
                    "no_classification": "❌ No Classification",
                    "no_decomposition": "❌ No Decomposition",
                    "no_retrieval": "❌ No Retrieval"
                }
                display_name = label_map.get(variant_name, variant_name.replace("_", " ").title())
                
                ablation_metrics.append({
                    "Variant": display_name,
                    "MRR": overall.get("mrr", 0.0),
                    "MPR (%)": overall.get("mpr", 0.0),
                    "Questions": overall.get("questions", 0)
                })
        
        if ablation_metrics:
            # Sort to show full system first
            ablation_metrics.sort(key=lambda x: (0 if "Full" in x["Variant"] else 1, x["Variant"]))
            
            df = pd.DataFrame(ablation_metrics)
            st.dataframe(df.style.format({"MRR": "{:.4f}", "MPR (%)": "{:.2f}"}), width='stretch')
            
            # Display metrics in columns
            if len(ablation_metrics) > 0:
                cols = st.columns(len(ablation_metrics))
                for i, metric in enumerate(ablation_metrics):
                    with cols[i]:
                        # Calculate delta from full system if available
                        if "Full" in metric["Variant"]:
                            st.metric(
                                metric["Variant"],
                                f"{metric['MRR']:.4f}",
                                f"{metric['Questions']} questions"
                            )
                        else:
                            full_mrr = next((m["MRR"] for m in ablation_metrics if "Full" in m["Variant"]), metric["MRR"])
                            delta = metric["MRR"] - full_mrr
                            st.metric(
                                metric["Variant"],
                                f"{metric['MRR']:.4f}",
                                f"{delta:+.4f} vs Full",
                                delta_color="inverse"
                            )
            
            # Key insights
            st.markdown("---")
            st.markdown("#### 📊 Component Impact Analysis")
            
            full_mrr = next((m["MRR"] for m in ablation_metrics if "Full" in m["Variant"]), None)
            
            if full_mrr is not None:
                st.markdown(f"**Baseline (Full System)**: MRR = {full_mrr:.4f}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    no_class = next((m for m in ablation_metrics if "Classification" in m["Variant"]), None)
                    if no_class:
                        impact = ((full_mrr - no_class["MRR"]) / full_mrr) * 100
                        st.metric(
                            "Classification Impact",
                            f"{impact:.1f}%",
                            f"MRR drops to {no_class['MRR']:.4f}"
                        )
                
                with col2:
                    no_decomp = next((m for m in ablation_metrics if "Decomposition" in m["Variant"]), None)
                    if no_decomp:
                        impact = ((full_mrr - no_decomp["MRR"]) / full_mrr) * 100
                        st.metric(
                            "Decomposition Impact",
                            f"{impact:.1f}%",
                            f"MRR drops to {no_decomp['MRR']:.4f}"
                        )
                
                with col3:
                    no_retrieval = next((m for m in ablation_metrics if "Retrieval" in m["Variant"]), None)
                    if no_retrieval:
                        impact = ((full_mrr - no_retrieval["MRR"]) / full_mrr) * 100
                        st.metric(
                            "Retrieval Impact",
                            f"{impact:.1f}%",
                            f"MRR drops to {no_retrieval['MRR']:.4f}"
                        )
                
                st.markdown("""
                **Interpretation**:
                - Higher percentage = Component is more critical to system performance
                - Retrieval typically has the largest impact (evidence-based answering)
                - Classification and Decomposition help optimize retrieval strategy
                """)
    else:
        st.info("⚠️ No ablation results found. Run: `python scripts/run_ablation_study.py`")


if __name__ == "__main__":
    main()
