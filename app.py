#!/usr/bin/env python3
"""
Typed-RAG Streamlit Demo App

Interactive UI to demonstrate:
1. Query interface with system selection
2. Typed-RAG pipeline visualization
"""

import streamlit as st
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from typed_rag.rag_system import DataType, ask_typed_question
from typed_rag.classifier import classify_question
from typed_rag.decompose import decompose_question
from typed_rag.core.keys import get_fastest_model
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
</style>
""", unsafe_allow_html=True)


def generate_llm_only_answer(question: str, model_name: str) -> tuple[str, float]:
    """Generate answer using only LLM without retrieval."""
    start = time.time()
    
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
    
    # Extract retrieved docs (simplified)
    docs = []
    if hasattr(result, 'aspects') and result.aspects:
        for aspect in result.aspects[:1]:  # Just first aspect for baseline
            if 'evidence' in aspect:
                docs.extend(aspect['evidence'][:5])
    
    return result.answer, elapsed, docs


def main():
    # Header
    st.markdown('<div class="main-header">🔍 Typed-RAG Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Aspect Question Answering with Type-Aware Decomposition</div>', unsafe_allow_html=True)
    
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
            ["gemini-2.0-flash-lite"],
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
        submit = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    
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
                        st.markdown(f'<div class="doc-card"><strong>Doc {i}:</strong> {doc.get("title", "Untitled")}<br><small>{doc.get("text", "")[:200]}...</small></div>', unsafe_allow_html=True)
            
            elif system == "Typed-RAG":
                st.markdown("### 🔬 Typed-RAG Pipeline")
                
                # Step 1: Classification
                with st.expander("📋 Step 1: Question Classification", expanded=True):
                    with st.spinner("Classifying question type..."):
                        q_type = classify_question(question, use_llm=True)
                    
                    st.markdown(f'<div class="pipeline-step"><strong>Question:</strong> {question}<br><strong>Classified Type:</strong> <span style="color: #1f77b4; font-weight: bold;">{q_type}</span></div>', unsafe_allow_html=True)
                    
                    type_descriptions = {
                        "EVIDENCE": "Factual question requiring evidence-based answer",
                        "COMPARISON": "Comparing multiple entities or concepts",
                        "REASON": "Asking for causes, reasons, or explanations",
                        "INSTRUCTION": "How-to or procedural question",
                        "EXPERIENCE": "Subjective or experiential question",
                        "DEBATE": "Opinion or argumentative question"
                    }
                    st.info(f"ℹ️ {type_descriptions.get(q_type, 'Unknown type')}")
                
                # Step 2: Decomposition
                with st.expander("🔀 Step 2: Multi-Aspect Decomposition", expanded=True):
                    with st.spinner("Decomposing into sub-questions..."):
                        decomposition = decompose_question(question, q_type)
                        aspects = decomposition.aspects if hasattr(decomposition, 'aspects') else []
                    
                    st.success(f"✓ Decomposed into {len(aspects)} aspects")
                    
                    for i, aspect in enumerate(aspects, 1):
                        st.markdown(f'<div class="aspect-card"><strong>Aspect {i}:</strong> {aspect.get("aspect", "Unknown")}<br><em>{aspect.get("sub_query", "")}</em></div>', unsafe_allow_html=True)
                
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
                                save_artifacts=False
                            )
                            elapsed = time.time() - start
                        except Exception as e:
                            if "429" in str(e) or "quota" in str(e).lower():
                                st.error("⚠️ **API Quota Exceeded**: You've reached the daily limit of 200 requests for Gemini API. Please try again tomorrow or use a different API key.")
                                st.stop()
                            else:
                                st.error(f"❌ Error: {str(e)}")
                                st.stop()
                    
                    if result.aspects:
                        st.success(f"✓ Retrieved and generated answers for {len(result.aspects)} aspects")
                        
                        for i, aspect in enumerate(result.aspects, 1):
                            with st.container():
                                st.markdown(f"**Aspect {i}: {aspect.get('aspect', 'Unknown')}**")
                                
                                # Show retrieved docs for this aspect
                                if 'evidence' in aspect:
                                    num_docs = len(aspect['evidence'])
                                    with st.expander(f"📄 {num_docs} documents retrieved", expanded=False):
                                        for j, doc in enumerate(aspect['evidence'][:3], 1):
                                            st.markdown(f'<div class="doc-card"><small><strong>Doc {j}:</strong> {doc.get("title", "")}<br>{doc.get("text", "")[:150]}...</small></div>', unsafe_allow_html=True)
                                
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
                    total_docs = sum(len(a.get('evidence', [])) for a in result.aspects) if result.aspects else 0
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


if __name__ == "__main__":
    main()
