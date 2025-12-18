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
    
    # Load linkage evaluation
    linkage_path = results_dir / "linkage_evaluation.json"
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
    tab1, tab2, tab3 = st.tabs(["🔍 Query Interface", "📊 Evaluation Results", "🔬 Ablation Study"])
    
    # Tab 1: Query Interface
    with tab1:
        _render_query_interface()
    
    # Tab 2: Evaluation Results
    with tab2:
        _render_evaluation_results()
    
    # Tab 3: Ablation Study
    with tab3:
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
            ["gemini-2.5-flash", "groq/llama-3.3-70b-versatile"],
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


def _render_evaluation_results():
    """Render evaluation metrics display."""
    st.markdown("### 📊 Evaluation Results")
    st.markdown("System performance metrics from evaluation runs.")
    
    repo_root = Path(__file__).parent
    eval_data = load_evaluation_results(repo_root)
    
    if not eval_data:
        st.info("No evaluation results found. Run evaluation scripts to generate results.")
        return
    
    # Display linkage evaluation
    if "linkage" in eval_data:
        st.markdown("#### Linkage Evaluation (MRR & MPR)")
        linkage = eval_data["linkage"]
        
        # Create comparison table
        systems = []
        metrics_data = []
        
        for system_name, system_data in linkage.items():
            if isinstance(system_data, dict) and "overall" in system_data:
                overall = system_data["overall"]
                systems.append(system_name.replace("_", " ").title())
                metrics_data.append({
                    "System": system_name.replace("_", " ").title(),
                    "MRR": overall.get("mrr", 0.0),
                    "MPR": overall.get("mpr", 0.0),
                    "Questions": overall.get("questions", 0)
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, width='stretch')
            
            # Display metrics in columns
            if len(metrics_data) > 0:
                cols = st.columns(len(metrics_data))
                for i, metric in enumerate(metrics_data):
                    with cols[i]:
                        st.metric(
                            metric["System"],
                            f"MRR: {metric['MRR']:.3f}",
                            f"MPR: {metric['MPR']:.1f}%"
                        )
    
    # Display classifier evaluation
    if "classifier" in eval_data:
        st.markdown("#### Classifier Evaluation")
        classifier = eval_data["classifier"]
        
        if "pattern_only" in classifier:
            pattern_data = classifier["pattern_only"]
            if "accuracy" in pattern_data:
                st.metric("Pattern-Based Accuracy", f"{pattern_data['accuracy']:.1%}")
            
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
                st.dataframe(type_df, width='stretch')
    
    # Display other evaluations
    for eval_key in ["decomposition", "generation", "retrieval"]:
        if eval_key in eval_data:
            st.markdown(f"#### {eval_key.title()} Evaluation")
            st.json(eval_data[eval_key])


def _render_ablation_study():
    """Render ablation study visualization."""
    st.markdown("### 🔬 Ablation Study Results")
    st.markdown("Comparison of Typed-RAG variants to evaluate component impact.")
    
    repo_root = Path(__file__).parent
    ablation_data = load_ablation_results(repo_root)
    
    if not ablation_data:
        st.info("No ablation study results found. Run ablation study script to generate results.")
        return
    
    # Display summary
    if "summary" in ablation_data:
        st.markdown("#### Performance Summary")
        summary = ablation_data["summary"]
        
        variants = []
        metrics = []
        
        if "variants" in summary:
            for variant_name, variant_data in summary["variants"].items():
                variants.append(variant_name.replace("_", " ").title())
                metrics.append({
                    "Variant": variant_name.replace("_", " ").title(),
                    "Success Rate": f"{variant_data.get('successful', 0)}/{variant_data.get('total', 0)}",
                    "Success %": f"{(variant_data.get('successful', 0) / max(variant_data.get('total', 1), 1)) * 100:.1f}%",
                    "Avg Latency (s)": variant_data.get("avg_latency_seconds", 0.0)
                })
        
        if metrics:
            df = pd.DataFrame(metrics)
            st.dataframe(df, width='stretch')
            
            # Visual comparison
            cols = st.columns(len(metrics))
            for i, metric in enumerate(metrics):
                with cols[i]:
                    st.metric(
                        metric["Variant"],
                        f"{metric['Avg Latency (s)']:.2f}s",
                        metric["Success Rate"]
                    )
    
    # Display linkage evaluation for ablation
    if "linkage" in ablation_data:
        st.markdown("#### Quality Metrics (MRR & MPR)")
        linkage = ablation_data["linkage"]
        
        ablation_metrics = []
        for variant_name, variant_data in linkage.items():
            if isinstance(variant_data, dict) and "overall" in variant_data:
                overall = variant_data["overall"]
                ablation_metrics.append({
                    "Variant": variant_name.replace("_", " ").title(),
                    "MRR": overall.get("mrr", 0.0),
                    "MPR": overall.get("mpr", 0.0),
                    "Questions": overall.get("questions", 0)
                })
        
        if ablation_metrics:
            df = pd.DataFrame(ablation_metrics)
            st.dataframe(df, width='stretch')
            
            # Display metrics in columns
            if len(ablation_metrics) > 0:
                cols = st.columns(len(ablation_metrics))
                for i, metric in enumerate(ablation_metrics):
                    with cols[i]:
                        st.metric(
                            metric["Variant"],
                            f"MRR: {metric['MRR']:.3f}",
                            f"MPR: {metric['MPR']:.1f}%"
                        )
            
            # Key insights
            st.markdown("#### Key Insights")
            if len(ablation_metrics) >= 4:
                full_mrr = next((m["MRR"] for m in ablation_metrics if m["Variant"] == "Full"), 0.0)
                no_decomp_mrr = next((m["MRR"] for m in ablation_metrics if m["Variant"] == "No Decomposition"), 0.0)
                
                if full_mrr > 0 and no_decomp_mrr > 0:
                    improvement = ((full_mrr - no_decomp_mrr) / no_decomp_mrr) * 100
                    st.info(f"**Decomposition Impact**: Full system shows {improvement:.1f}% improvement in MRR over no-decomposition variant.")


if __name__ == "__main__":
    main()
