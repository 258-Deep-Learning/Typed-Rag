# scripts/query.py
import json
import sys
from pathlib import Path
import typer
from typing import Literal

# Add parent directory to path to import retrieval module
sys.path.append(str(Path(__file__).parent.parent))
from retrieval.hybrid import HybridRetriever

app = typer.Typer()

@app.command("search")
def search_cmd(
    q: str = typer.Option(..., "--q", help="Query string to search for"),
    k: int = typer.Option(5, "--k", help="Number of results to return"),
    mode: Literal["bm25", "faiss", "hybrid"] = typer.Option("hybrid", "--mode", help="Search mode: bm25, faiss, or hybrid"),
    bm25_path: str = typer.Option("indexes/bm25_rank.pkl", "--bm25-path", help="Path to BM25 index"),
    faiss_dir: str = typer.Option("indexes/faiss_bge_small", "--faiss-dir", help="Path to FAISS index directory"), 
    meta_path: str = typer.Option("indexes/meta.jsonl", "--meta-path", help="Path to metadata file"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty print JSON output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output with component health")
):
    """
    Query the hybrid retrieval system.
    
    Examples:
        python scripts/query.py --q "Who discovered penicillin?" --k 3 --mode hybrid
        python scripts/query.py --q "machine learning" --k 5 --mode bm25 --pretty
    """
    
    try:
        # Initialize retriever
        if verbose:
            typer.echo("🔧 Initializing retriever...", err=True)
        
        retriever = HybridRetriever(
            bm25_path=bm25_path,
            faiss_dir=faiss_dir, 
            meta_path=meta_path
        )
        
        # Health check
        if verbose:
            health = retriever.health_check()
            typer.echo("📊 Component Health:", err=True)
            for component, status in health.items():
                status_icon = "✅" if status else "❌"
                typer.echo(f"  {status_icon} {component}: {status}", err=True)
            typer.echo("", err=True)
        
        # Perform search
        if verbose:
            typer.echo(f"🔍 Searching for: '{q}' (mode: {mode}, k: {k})", err=True)
        
        results = retriever.retrieve(query=q, k=k, mode=mode)
        
        # Output results
        if pretty:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(results, ensure_ascii=False))
        
        # Verbose summary
        if verbose:
            typer.echo(f"📈 Found {len(results)} results", err=True)
            if results:
                typer.echo("Top result:", err=True)
                top = results[0]
                typer.echo(f"  📄 {top.get('title', 'No title')}", err=True)
                typer.echo(f"  🔗 {top.get('url', 'No URL')}", err=True)
                typer.echo(f"  📊 Score: {top.get('score', 0):.4f}", err=True)
                if mode == "hybrid":
                    typer.echo(f"      BM25: {top.get('bm25_score', 0):.4f}, FAISS: {top.get('faiss_score', 0):.4f}", err=True)
    
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command()
def health():
    """Check the health of all retrieval components."""
    try:
        retriever = HybridRetriever()
        health = retriever.health_check()
        
        typer.echo("🏥 Retrieval System Health Check")
        typer.echo("=" * 40)
        
        all_healthy = True
        for component, status in health.items():
            status_icon = "✅" if status else "❌"
            typer.echo(f"{status_icon} {component.replace('_', ' ').title()}: {status}")
            if not status:
                all_healthy = False
        
        typer.echo("=" * 40)
        if all_healthy:
            typer.echo("🎉 All systems operational!")
        else:
            typer.echo("⚠️  Some components have issues. Check the logs above.")
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Health check failed: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command() 
def demo():
    """Run a quick demo with sample queries."""
    sample_queries = [
        "What is machine learning?",
        "How does artificial intelligence work?", 
        "Python programming tutorial",
        "data science methods"
    ]
    
    try:
        retriever = HybridRetriever()
        
        typer.echo("🚀 Running Demo Queries")
        typer.echo("=" * 50)
        
        for i, query in enumerate(sample_queries, 1):
            typer.echo(f"\n📝 Query {i}: {query}")
            typer.echo("-" * 30)
            
            results = retriever.retrieve(query, k=2, mode="hybrid")
            
            if results:
                for j, result in enumerate(results, 1):
                    title = result.get('title', 'No title')[:50] + "..." if len(result.get('title', '')) > 50 else result.get('title', 'No title')
                    score = result.get('score', 0)
                    typer.echo(f"  {j}. {title} (score: {score:.3f})")
            else:
                typer.echo("  No results found")
        
        typer.echo("\n🎉 Demo completed!")
        
    except Exception as e:
        typer.echo(f"❌ Demo failed: {str(e)}", err=True)
        raise typer.Exit(1)

# Make search the default command
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    q: str = typer.Option(None, "--q", help="Query string to search for"),
    k: int = typer.Option(5, "--k", help="Number of results to return"),
    mode: Literal["bm25", "faiss", "hybrid"] = typer.Option("hybrid", "--mode", help="Search mode: bm25, faiss, or hybrid"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty print JSON output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output with component health")
):
    """Query the hybrid retrieval system (default command)."""
    if ctx.invoked_subcommand is None:
        if q is None:
            typer.echo("❌ Query required. Use --q 'your query' or run a subcommand.", err=True)
            raise typer.Exit(1)
        
        # Call the search function directly
        search_cmd(q=q, k=k, mode=mode, pretty=pretty, verbose=verbose,
                  bm25_path="indexes/bm25_rank.pkl",
                  faiss_dir="indexes/faiss_bge_small", 
                  meta_path="indexes/meta.jsonl")

if __name__ == "__main__":
    app()
