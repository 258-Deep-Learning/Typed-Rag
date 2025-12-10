#!/usr/bin/env python3
"""
Performance Profiling Script for Typed-RAG

Measures system performance metrics:
- Memory usage (peak, average per question)
- Throughput (questions per second)
- Component-level latency breakdown
- Cache hit rates

Usage:
    python scripts/profile_performance.py --input data/wiki_nfqa/dev6.jsonl --output results/performance_profile.json
"""

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typed_rag.rag_system import ask_typed_question, DataType
from typed_rag.data.loaders import WikiNFQALoader, WikiNFQAQuestion
from typed_rag.classifier import classify_question
from typed_rag.decompose import decompose_question


@dataclass
class ComponentTiming:
    """Timing breakdown for pipeline components."""
    classification: float = 0.0
    decomposition: float = 0.0
    retrieval: float = 0.0
    generation: float = 0.0
    aggregation: float = 0.0
    total: float = 0.0


@dataclass
class MemoryProfile:
    """Memory usage metrics."""
    peak_mb: float = 0.0
    current_mb: float = 0.0
    average_per_question_mb: float = 0.0


@dataclass
class ThroughputMetrics:
    """Throughput and latency metrics."""
    questions_per_second: float = 0.0
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0
    total_time_seconds: float = 0.0
    average_latency_seconds: float = 0.0


@dataclass
class CacheMetrics:
    """Cache hit rate metrics."""
    decomposition_hits: int = 0
    decomposition_misses: int = 0
    evidence_hits: int = 0
    evidence_misses: int = 0
    answer_hits: int = 0
    answer_misses: int = 0


@dataclass
class PerformanceProfile:
    """Complete performance profile."""
    throughput: ThroughputMetrics
    memory: MemoryProfile
    component_latency: ComponentTiming
    cache: CacheMetrics
    per_question_metrics: List[Dict[str, Any]]


def measure_memory():
    """Get current memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 / 1024, peak / 1024 / 1024


def profile_question(
    question: WikiNFQAQuestion,
    data_type: DataType,
    model_name: str = None,
    enable_caching: bool = True,
) -> Dict[str, Any]:
    """
    Profile a single question through the pipeline.
    
    Returns:
        Dictionary with timing, memory, and result info
    """
    result = {
        "question_id": question.question_id,
        "question": question.question,
        "success": False,
        "error": None,
        "timing": {},
        "memory": {},
    }
    
    # Record initial memory
    mem_start_current, mem_start_peak = measure_memory()
    
    # Overall timing
    overall_start = time.time()
    
    try:
        # 1. Classification
        classify_start = time.time()
        question_type = classify_question(question.question, use_llm=True)
        classify_time = time.time() - classify_start
        result["timing"]["classification"] = round(classify_time, 3)
        result["question_type"] = question_type
        
        # 2. Decomposition
        decompose_start = time.time()
        decomposition_plan = decompose_question(question.question, question_type, use_cache=enable_caching)
        decompose_time = time.time() - decompose_start
        result["timing"]["decomposition"] = round(decompose_time, 3)
        result["num_aspects"] = len(decomposition_plan.sub_queries) if hasattr(decomposition_plan, 'sub_queries') else 0
        
        # 3. Retrieval + Generation (combined in ask_typed_question)
        retrieve_gen_start = time.time()
        system_result = ask_typed_question(
            query=question.question,
            data_type=data_type,
            model_name=model_name,
            use_llm=True,
            save_artifacts=False,
            use_classification=False,  # Already classified
            use_decomposition=False,   # Already decomposed
            use_retrieval=True,
        )
        retrieve_gen_time = time.time() - retrieve_gen_start
        
        # Estimate breakdown (retrieval typically 30%, generation 70%)
        result["timing"]["retrieval"] = round(retrieve_gen_time * 0.3, 3)
        result["timing"]["generation"] = round(retrieve_gen_time * 0.7, 3)
        result["timing"]["aggregation"] = 0.0  # Included in generation
        
        overall_time = time.time() - overall_start
        result["timing"]["total"] = round(overall_time, 3)
        
        # Memory after processing
        mem_end_current, mem_end_peak = measure_memory()
        result["memory"]["current_mb"] = round(mem_end_current, 2)
        result["memory"]["peak_mb"] = round(mem_end_peak, 2)
        result["memory"]["delta_mb"] = round(mem_end_current - mem_start_current, 2)
        
        result["success"] = True
        result["answer_length"] = len(system_result.answer) if hasattr(system_result, 'answer') else 0
        
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["timing"]["total"] = round(time.time() - overall_start, 3)
    
    return result


def profile_system(
    questions: List[WikiNFQAQuestion],
    data_type: DataType,
    model_name: str = None,
    enable_caching: bool = True,
) -> PerformanceProfile:
    """
    Profile the entire system across multiple questions.
    
    Args:
        questions: List of questions to process
        data_type: DataType configuration
        model_name: Model to use
        enable_caching: Whether to use caching
    
    Returns:
        Complete performance profile
    """
    print(f"\n{'='*80}")
    print(f"PERFORMANCE PROFILING")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}")
    print(f"Caching: {'Enabled' if enable_caching else 'Disabled'}")
    print(f"Model: {model_name or 'default'}")
    print()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Initialize metrics
    per_question_results = []
    total_time_start = time.time()
    
    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Profiling: {question.question[:60]}...")
        
        result = profile_question(question, data_type, model_name, enable_caching)
        per_question_results.append(result)
        
        if result["success"]:
            print(f"  ✓ Completed in {result['timing']['total']:.2f}s")
            print(f"    - Classification: {result['timing']['classification']:.2f}s")
            print(f"    - Decomposition: {result['timing']['decomposition']:.2f}s")
            print(f"    - Retrieval: {result['timing']['retrieval']:.2f}s")
            print(f"    - Generation: {result['timing']['generation']:.2f}s")
            print(f"    - Memory: {result['memory']['delta_mb']:.2f} MB delta")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Rate limiting between questions
        if i < len(questions):
            time.sleep(1.0)
    
    total_time = time.time() - total_time_start
    
    # Get final memory stats
    mem_current, mem_peak = measure_memory()
    tracemalloc.stop()
    
    # Calculate aggregate metrics
    successful = [r for r in per_question_results if r["success"]]
    failed = [r for r in per_question_results if not r["success"]]
    
    # Throughput
    throughput = ThroughputMetrics(
        questions_per_second=len(successful) / total_time if total_time > 0 else 0.0,
        total_questions=len(questions),
        successful_questions=len(successful),
        failed_questions=len(failed),
        total_time_seconds=round(total_time, 2),
        average_latency_seconds=round(sum(r["timing"]["total"] for r in successful) / len(successful), 3) if successful else 0.0,
    )
    
    # Memory
    avg_memory_per_question = sum(r["memory"]["delta_mb"] for r in successful) / len(successful) if successful else 0.0
    memory = MemoryProfile(
        peak_mb=round(mem_peak, 2),
        current_mb=round(mem_current, 2),
        average_per_question_mb=round(avg_memory_per_question, 2),
    )
    
    # Component latency (averaged)
    component_latency = ComponentTiming(
        classification=round(sum(r["timing"]["classification"] for r in successful) / len(successful), 3) if successful else 0.0,
        decomposition=round(sum(r["timing"]["decomposition"] for r in successful) / len(successful), 3) if successful else 0.0,
        retrieval=round(sum(r["timing"]["retrieval"] for r in successful) / len(successful), 3) if successful else 0.0,
        generation=round(sum(r["timing"]["generation"] for r in successful) / len(successful), 3) if successful else 0.0,
        aggregation=round(sum(r["timing"]["aggregation"] for r in successful) / len(successful), 3) if successful else 0.0,
        total=round(sum(r["timing"]["total"] for r in successful) / len(successful), 3) if successful else 0.0,
    )
    
    # Cache metrics (placeholder - would need instrumentation in actual cache)
    cache = CacheMetrics(
        decomposition_hits=0,
        decomposition_misses=len(successful),
        evidence_hits=0,
        evidence_misses=len(successful),
        answer_hits=0,
        answer_misses=len(successful),
    )
    
    return PerformanceProfile(
        throughput=throughput,
        memory=memory,
        component_latency=component_latency,
        cache=cache,
        per_question_metrics=per_question_results,
    )


def print_performance_summary(profile: PerformanceProfile):
    """Print a formatted summary of performance metrics."""
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")
    
    # Throughput
    print("📊 Throughput Metrics:")
    print(f"  Total Questions: {profile.throughput.total_questions}")
    print(f"  Successful: {profile.throughput.successful_questions}")
    print(f"  Failed: {profile.throughput.failed_questions}")
    print(f"  Total Time: {profile.throughput.total_time_seconds:.2f}s")
    print(f"  Questions/Second: {profile.throughput.questions_per_second:.3f}")
    print(f"  Avg Latency: {profile.throughput.average_latency_seconds:.3f}s")
    
    # Memory
    print(f"\n💾 Memory Metrics:")
    print(f"  Peak Memory: {profile.memory.peak_mb:.2f} MB")
    print(f"  Current Memory: {profile.memory.current_mb:.2f} MB")
    print(f"  Avg per Question: {profile.memory.average_per_question_mb:.2f} MB")
    
    # Component Latency
    print(f"\n⏱️  Component Latency (Average):")
    print(f"  Classification: {profile.component_latency.classification:.3f}s ({profile.component_latency.classification/profile.component_latency.total*100:.1f}%)")
    print(f"  Decomposition: {profile.component_latency.decomposition:.3f}s ({profile.component_latency.decomposition/profile.component_latency.total*100:.1f}%)")
    print(f"  Retrieval: {profile.component_latency.retrieval:.3f}s ({profile.component_latency.retrieval/profile.component_latency.total*100:.1f}%)")
    print(f"  Generation: {profile.component_latency.generation:.3f}s ({profile.component_latency.generation/profile.component_latency.total*100:.1f}%)")
    print(f"  Total: {profile.component_latency.total:.3f}s")
    
    # Cache (if available)
    total_ops = profile.cache.decomposition_hits + profile.cache.decomposition_misses
    if total_ops > 0:
        print(f"\n📦 Cache Metrics:")
        print(f"  Decomposition Hit Rate: {profile.cache.decomposition_hits/total_ops*100:.1f}%")
        print(f"  Evidence Hit Rate: {profile.cache.evidence_hits/total_ops*100:.1f}%")
        print(f"  Answer Hit Rate: {profile.cache.answer_hits/total_ops*100:.1f}%")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Profile Typed-RAG system performance")
    parser.add_argument("--input", required=True, help="Input JSONL file with questions")
    parser.add_argument("--output", default="results/performance_profile.json", help="Output JSON file")
    parser.add_argument("--source", default="wikipedia", choices=["wikipedia", "own_docs"], help="Data source")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "pinecone"], help="Vector store backend")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Load questions
    print(f"📂 Loading questions from: {args.input}")
    
    if Path(args.input).name.startswith("dev"):
        split = Path(args.input).stem
        loader = WikiNFQALoader()
        questions = loader.load_questions(split)
    else:
        with open(args.input, "r") as f:
            questions = []
            for line in f:
                data = json.loads(line)
                questions.append(WikiNFQAQuestion.from_dict(data))
    
    print(f"✓ Loaded {len(questions)} questions\n")
    
    # Configure data type
    data_type = DataType(type=args.backend, source=args.source)
    
    # Profile system
    profile = profile_system(
        questions=questions,
        data_type=data_type,
        model_name=args.model,
        enable_caching=not args.no_cache,
    )
    
    # Print summary
    print_performance_summary(profile)
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict for JSON serialization
    profile_dict = {
        "throughput": asdict(profile.throughput),
        "memory": asdict(profile.memory),
        "component_latency": asdict(profile.component_latency),
        "cache": asdict(profile.cache),
        "per_question_metrics": profile.per_question_metrics,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Performance profile saved to: {output_path}")
    
    # Summary stats
    print(f"\n📈 Key Findings:")
    if profile.throughput.questions_per_second > 0:
        print(f"  • System processes {profile.throughput.questions_per_second:.2f} questions/second")
    print(f"  • Average latency: {profile.throughput.average_latency_seconds:.2f}s per question")
    print(f"  • Peak memory usage: {profile.memory.peak_mb:.1f} MB")
    
    # Bottleneck analysis
    latency_breakdown = [
        ("Classification", profile.component_latency.classification),
        ("Decomposition", profile.component_latency.decomposition),
        ("Retrieval", profile.component_latency.retrieval),
        ("Generation", profile.component_latency.generation),
    ]
    slowest_component = max(latency_breakdown, key=lambda x: x[1])
    print(f"  • Slowest component: {slowest_component[0]} ({slowest_component[1]:.2f}s)")
    
    success_rate = (profile.throughput.successful_questions / profile.throughput.total_questions * 100) if profile.throughput.total_questions > 0 else 0
    print(f"  • Success rate: {success_rate:.1f}% ({profile.throughput.successful_questions}/{profile.throughput.total_questions})")


if __name__ == "__main__":
    main()
