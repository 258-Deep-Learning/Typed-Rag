#!/usr/bin/env python3
"""
Quick test to verify Gemini API key works.
"""

import os
import sys

def test_gemini():
    """Test Gemini API connection."""
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set!")
        print("\nPlease run:")
        print('  export GOOGLE_API_KEY="your-api-key-here"')
        print("\nGet your key at: https://aistudio.google.com/app/apikey")
        return False
    
    print(f"✓ GOOGLE_API_KEY is set (starts with: {api_key[:10]}...)")
    
    # Try to import and use Gemini
    try:
        import google.generativeai as genai
        print("✓ google-generativeai package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import google-generativeai: {e}")
        print("\nPlease install: pip install google-generativeai")
        return False
    
    # Configure and test
    try:
        genai.configure(api_key=api_key)
        print("✓ API key configured")
        
        # Create model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        print("✓ Model initialized (gemini-2.0-flash-exp)")
        
        # Test simple generation
        print("\n📝 Testing generation with a simple question...")
        response = model.generate_content("What is 2+2? Answer with just the number.")
        
        answer = response.text.strip()
        print(f"✓ Response received: '{answer}'")
        
        print("\n✅ Gemini is working perfectly!")
        print("\nYou're ready to run the RAG pipeline:")
        print("  ./quickstart.sh my_documents")
        print("\nOr test with a question:")
        print('  python3 typed_rag/scripts/run_llm_only.py --in data.jsonl --out results.jsonl')
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Gemini: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API key - get a new one at https://aistudio.google.com/app/apikey")
        print("  2. API quota exceeded - check your usage")
        print("  3. Network connection issue")
        return False


if __name__ == "__main__":
    success = test_gemini()
    sys.exit(0 if success else 1)

