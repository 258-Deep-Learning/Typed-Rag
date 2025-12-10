#!/usr/bin/env python3
"""
Evaluation Plots Generator for Typed-RAG

Generates publication-ready plots from evaluation results:
- Ablation study latency comparison
- Ablation study success rates
- MRR/MPR comparison across systems
- Classifier performance (Precision/Recall/F1)
- Confusion matrix heatmap

Usage:
    python scripts/create_evaluation_plots.py
    python scripts/create_evaluation_plots.py --output results/plots --format png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for publication-ready plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file and return data."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {file_path}: {e}")
        return {}


def plot_ablation_latency(data: Dict[str, Any], output_dir: Path, format: str = 'png'):
    """Generate bar chart comparing latency across ablation variants."""
    if not data or 'variants' not in data:
        print("Skipping ablation latency plot: No data available")
        return
    
    variants = data['variants']
    
    # Prepare data
    variant_names = []
    latencies = []
    
    for variant_key, variant_data in variants.items():
        if variant_data['successful'] > 0:  # Only include successful variants
            variant_names.append(variant_key.replace('_', ' ').title())
            latencies.append(variant_data['avg_latency_seconds'])
    
    if not variant_names:
        print("Skipping ablation latency plot: No successful variants")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = sns.color_palette("husl", len(variant_names))
    bars = ax.bar(variant_names, latencies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Ablation Variant', fontweight='bold')
    ax.set_ylabel('Average Latency (seconds)', fontweight='bold')
    ax.set_title('Ablation Study: Latency Comparison', fontweight='bold', pad=15)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    # Rotate x labels if needed
    if len(variant_names) > 3:
        plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'ablation_latency.{format}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()


def plot_ablation_success(data: Dict[str, Any], output_dir: Path, format: str = 'png'):
    """Generate grouped bar chart for ablation success rates."""
    if not data or 'variants' not in data:
        print("Skipping ablation success plot: No data available")
        return
    
    variants = data['variants']
    
    # Prepare data
    variant_names = []
    successful = []
    failed = []
    
    for variant_key, variant_data in variants.items():
        variant_names.append(variant_key.replace('_', ' ').title())
        successful.append(variant_data['successful'])
        failed.append(variant_data['failed'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(variant_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, successful, width, label='Successful', 
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, failed, width, label='Failed', 
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Ablation Variant', fontweight='bold')
    ax.set_ylabel('Number of Questions', fontweight='bold')
    ax.set_title('Ablation Study: Success vs Failed Questions', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names)
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8)
    
    # Rotate x labels if needed
    if len(variant_names) > 3:
        plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'ablation_success.{format}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()


def plot_mrr_mpr_comparison(linkage_data: Dict[str, Any], output_dir: Path, format: str = 'png'):
    """Generate bar chart comparing MRR and MPR across systems."""
    if not linkage_data:
        print("Skipping MRR/MPR plot: No linkage data available")
        return
    
    # Prepare data
    systems = []
    mrr_scores = []
    mpr_scores = []
    
    for system_name, system_data in linkage_data.items():
        if isinstance(system_data, dict) and 'overall' in system_data:
            overall = system_data['overall']
            if overall.get('questions', 0) > 0:  # Only include systems with data
                systems.append(system_name.replace('_', ' ').title())
                mrr_scores.append(overall.get('mrr', 0.0))
                mpr_scores.append(overall.get('mpr', 0.0))
    
    if not systems:
        print("Skipping MRR/MPR plot: No systems with evaluation data")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = sns.color_palette("Set2", len(systems))
    
    # MRR subplot
    bars1 = ax1.bar(systems, mrr_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('System', fontweight='bold')
    ax1.set_ylabel('MRR Score', fontweight='bold')
    ax1.set_title('Mean Reciprocal Rank (MRR)', fontweight='bold', pad=15)
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    if len(systems) > 3:
        ax1.tick_params(axis='x', rotation=15)
    
    # MPR subplot
    bars2 = ax2.bar(systems, mpr_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('System', fontweight='bold')
    ax2.set_ylabel('MPR Score (%)', fontweight='bold')
    ax2.set_title('Mean Percentile Rank (MPR)', fontweight='bold', pad=15)
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    if len(systems) > 3:
        ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'mrr_mpr_comparison.{format}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()


def plot_classifier_performance(classifier_data: Dict[str, Any], output_dir: Path, format: str = 'png'):
    """Generate bar chart for classifier performance per question type."""
    if not classifier_data:
        print("Skipping classifier performance plot: No data available")
        return
    
    # Use pattern_only results (more interesting with varied performance)
    method = 'pattern_only' if 'pattern_only' in classifier_data else 'gemini'
    
    if method not in classifier_data or 'per_type' not in classifier_data[method]:
        print("Skipping classifier performance plot: No per-type data available")
        return
    
    per_type = classifier_data[method]['per_type']
    
    # Prepare data
    question_types = []
    precision = []
    recall = []
    f1 = []
    
    for qtype, metrics in per_type.items():
        question_types.append(qtype)
        precision.append(metrics.get('precision', 0.0))
        recall.append(metrics.get('recall', 0.0))
        f1.append(metrics.get('f1', 0.0))
    
    if not question_types:
        print("Skipping classifier performance plot: No question types found")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(question_types))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', 
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, recall, width, label='Recall', 
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                   color='#e67e22', edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Question Type', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    method_name = "Pattern-Based" if method == "pattern_only" else "Gemini"
    ax.set_title(f'Classifier Performance ({method_name}): Precision, Recall, F1', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(question_types, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'classifier_performance.{format}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()


def plot_confusion_matrix(classifier_data: Dict[str, Any], output_dir: Path, format: str = 'png'):
    """Generate confusion matrix heatmap for classifier."""
    if not classifier_data:
        print("Skipping confusion matrix: No data available")
        return
    
    # Use pattern_only for more interesting visualization
    method = 'pattern_only' if 'pattern_only' in classifier_data else 'gemini'
    
    if method not in classifier_data or 'confusion_matrix' not in classifier_data[method]:
        print("Skipping confusion matrix: No confusion matrix data available")
        return
    
    cm_data = classifier_data[method]['confusion_matrix']
    labels = cm_data.get('labels', [])
    matrix = cm_data.get('matrix', [])
    
    if not labels or not matrix:
        print("Skipping confusion matrix: Empty data")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert to numpy array
    cm_array = np.array(matrix)
    
    # Create heatmap
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5, linecolor='gray')
    
    # Customize plot
    ax.set_xlabel('Predicted Type', fontweight='bold', labelpad=10)
    ax.set_ylabel('True Type', fontweight='bold', labelpad=10)
    method_name = "Pattern-Based" if method == "pattern_only" else "Gemini"
    ax.set_title(f'Confusion Matrix ({method_name} Classifier)', fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'confusion_matrix.{format}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()


def plot_combined_ablation_metrics(ablation_summary: Dict[str, Any], 
                                   linkage_data: Dict[str, Any],
                                   output_dir: Path, format: str = 'png'):
    """Generate combined plot showing latency + success rate side-by-side."""
    if not ablation_summary or 'variants' not in ablation_summary:
        print("Skipping combined ablation metrics: No data available")
        return
    
    variants_data = ablation_summary['variants']
    
    # Filter out variants with no successful runs
    variant_names = []
    latencies = []
    success_rates = []
    
    for variant_key, variant_data in variants_data.items():
        if variant_data['total'] > 0:
            variant_names.append(variant_key.replace('_', ' ').title())
            latencies.append(variant_data['avg_latency_seconds'])
            success_rate = (variant_data['successful'] / variant_data['total']) * 100
            success_rates.append(success_rate)
    
    if not variant_names:
        print("Skipping combined ablation metrics: No valid variants")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = sns.color_palette("Set2", len(variant_names))
    
    # Latency subplot
    bars1 = ax1.bar(variant_names, latencies, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Variant', fontweight='bold')
    ax1.set_ylabel('Average Latency (seconds)', fontweight='bold')
    ax1.set_title('Latency by Variant', fontweight='bold', pad=15)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=8)
    
    if len(variant_names) > 3:
        ax1.tick_params(axis='x', rotation=15)
    
    # Success rate subplot
    bars2 = ax2.bar(variant_names, success_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Variant', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('Success Rate by Variant', fontweight='bold', pad=15)
    ax2.set_ylim(0, 105)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=8)
    
    if len(variant_names) > 3:
        ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f'ablation_combined.{format}'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Generated: {output_file}")
    plt.close()


def print_summary_statistics(ablation_summary: Dict[str, Any], 
                            classifier_data: Dict[str, Any],
                            linkage_data: Dict[str, Any]):
    """Print summary statistics from all evaluations."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY STATISTICS")
    print("="*80)
    
    # Ablation study summary
    if ablation_summary and 'variants' in ablation_summary:
        print("\n📊 Ablation Study:")
        print(f"  Total Questions: {ablation_summary.get('total_questions', 0)}")
        
        for variant_key, variant_data in ablation_summary['variants'].items():
            success_rate = (variant_data['successful'] / variant_data['total'] * 100) if variant_data['total'] > 0 else 0
            print(f"\n  {variant_key.replace('_', ' ').title()}:")
            print(f"    Success Rate: {success_rate:.1f}% ({variant_data['successful']}/{variant_data['total']})")
            print(f"    Avg Latency: {variant_data['avg_latency_seconds']:.2f}s")
    
    # Classifier summary
    if classifier_data:
        print("\n🎯 Classifier Performance:")
        for method in ['pattern_only', 'gemini']:
            if method in classifier_data:
                method_name = "Pattern-Based" if method == "pattern_only" else "Gemini"
                accuracy = classifier_data[method].get('accuracy', 0.0)
                print(f"  {method_name}: {accuracy:.1%} accuracy")
    
    # Linkage summary
    if linkage_data:
        print("\n📈 Quality Metrics (MRR/MPR):")
        for system_name, system_data in linkage_data.items():
            if isinstance(system_data, dict) and 'overall' in system_data:
                overall = system_data['overall']
                if overall.get('questions', 0) > 0:
                    print(f"  {system_name.replace('_', ' ').title()}:")
                    print(f"    MRR: {overall.get('mrr', 0.0):.3f}")
                    print(f"    MPR: {overall.get('mpr', 0.0):.1f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots for Typed-RAG")
    parser.add_argument("--results", default="results", help="Results directory (default: results)")
    parser.add_argument("--output", default="results/plots", help="Output directory for plots")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"], 
                       help="Output format (default: png)")
    
    args = parser.parse_args()
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / args.results
    output_dir = repo_root / args.output
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating evaluation plots...")
    print(f"   Results: {results_dir}")
    print(f"   Output:  {output_dir}")
    print(f"   Format:  {args.format}\n")
    
    # Load data
    ablation_summary = load_json(results_dir / "ablation" / "summary.json")
    ablation_linkage = load_json(results_dir / "ablation_linkage_evaluation.json")
    classifier_data = load_json(results_dir / "classifier_evaluation.json")
    linkage_data = load_json(results_dir / "linkage_evaluation.json")
    
    # Generate plots
    plot_ablation_latency(ablation_summary, output_dir, args.format)
    plot_ablation_success(ablation_summary, output_dir, args.format)
    plot_combined_ablation_metrics(ablation_summary, ablation_linkage, output_dir, args.format)
    
    # Use ablation linkage or general linkage data
    linkage_to_plot = ablation_linkage if ablation_linkage else linkage_data
    plot_mrr_mpr_comparison(linkage_to_plot, output_dir, args.format)
    
    plot_classifier_performance(classifier_data, output_dir, args.format)
    plot_confusion_matrix(classifier_data, output_dir, args.format)
    
    # Print summary statistics
    print_summary_statistics(ablation_summary, classifier_data, linkage_to_plot)
    
    print(f"\n✅ All plots generated successfully in: {output_dir}")
    print(f"\nGenerated files:")
    for plot_file in sorted(output_dir.glob(f"*.{args.format}")):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()
