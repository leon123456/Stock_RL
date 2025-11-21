from src.news_processor import NewsProcessor
import json

def test_news():
    print("=== Manual News Logic Test ===")
    print("Initializing NewsProcessor (this may take a moment to load env vars)...")
    
    try:
        np_proc = NewsProcessor()
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    print("\nEnter a news headline to test (or 'q' to quit):")
    
    while True:
        text = input("\n> ")
        if text.lower() in ['q', 'quit', 'exit']:
            break
            
        if not text.strip():
            continue
            
        print("\nAnalyzing with Qwen-max...")
        analysis = np_proc.analyze_news_with_qwen(text, symbol="Target Stock")
        
        if analysis:
            print("\n[Analysis Result]")
            print(json.dumps(analysis, indent=2, ensure_ascii=False))
            
            print("\nGenerating Embedding...")
            # Construct rich text manually to match process_daily_news logic
            rich_text = f"Sentiment: {analysis.get('sentiment_score', 0)}. Summary: {analysis.get('summary', '')}. Reasoning: {analysis.get('reasoning', '')}"
            emb = np_proc.get_embedding(rich_text)
            print(f"Embedding Shape: {emb.shape}")
            print(f"Embedding Sample (first 5): {emb[:5]}")
        else:
            print("Analysis failed.")

if __name__ == "__main__":
    test_news()
