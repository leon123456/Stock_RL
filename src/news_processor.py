import os
import json
import numpy as np
from http import HTTPStatus
import dashscope
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class NewsProcessor:
    """
    Process news text using Alibaba Cloud Qwen-max API.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("Warning: DASHSCOPE_API_KEY not found. News processing will fail.")
        
        dashscope.api_key = self.api_key

    def analyze_news_with_qwen(self, news_content, symbol="Stock"):
        """
        Uses Qwen-max to analyze the news and extract structured sentiment/summary.
        """
        prompt = f"""
        分析以下财经新闻对 {symbol} 股价的影响：
        新闻内容：{news_content}
        
        请输出 JSON 格式，不要包含 Markdown 格式（如 ```json ... ```），直接输出 JSON 字符串：
        {{
            "summary": "一句话摘要",
            "sentiment_score": -1到1之间的分数 (浮点数),
            "reasoning": "简短理由"
        }}
        """
        
        try:
            resp = dashscope.Generation.call(
                model='qwen-max',
                prompt=prompt,
                result_format='message',  # set the result to be "message" format.
            )
            
            if resp.status_code == HTTPStatus.OK:
                content = resp.output.choices[0].message.content
                # Clean up potential markdown code blocks
                content = content.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError:
                    print(f"JSON Parse Error. Raw content: {content}")
                    return None
            else:
                print(f"Qwen API Error: {resp.code} - {resp.message}")
                return None
                
        except Exception as e:
            print(f"Qwen Analysis Error: {e}")
            return None

    def get_embedding(self, text):
        """
        Uses DashScope Text Embedding API to vectorize text.
        """
        try:
            resp = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v1,
                input=text
            )
            
            if resp.status_code == HTTPStatus.OK:
                embedding = resp.output['embeddings'][0]['embedding']
                return np.array(embedding)
            else:
                print(f"Embedding API Error: {resp}")
                return np.zeros(1536)
                
        except Exception as e:
            print(f"Embedding Error: {e}")
            return np.zeros(1536)

    def process_daily_news(self, news_list, symbol="Stock"):
        """
        Process a list of news items for a single day.
        1. Analyze each news with Qwen-max.
        2. Embed the summary + sentiment + reasoning.
        3. Mean pool the embeddings.
        """
        if not news_list:
            return np.zeros(768)
            
        embeddings = []
        for news in news_list:
            # Step 1: Analyze with Qwen
            analysis = self.analyze_news_with_qwen(news, symbol)
            
            if analysis:
                # Construct a rich text representation for embedding
                # "Sentiment: 0.8. Summary: Revenue up. Reasoning: Strong growth."
                rich_text = f"Sentiment: {analysis.get('sentiment_score', 0)}. Summary: {analysis.get('summary', '')}. Reasoning: {analysis.get('reasoning', '')}"
                print(f"Analyzed News: {rich_text}")
            else:
                # Fallback to raw text if analysis fails
                rich_text = news
                print("Analysis failed, using raw text.")
            
            # Step 2: Embed
            emb = self.get_embedding(rich_text)
            embeddings.append(emb)
            
        # Mean pooling
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(768)

if __name__ == "__main__":
    # Test
    np_proc = NewsProcessor()
    text = "Tencent released its Q3 financial report. Revenue increased by 10% YoY, and net profit exceeded expectations."
    print(f"Testing with news: {text}")
    
    # Test single analysis
    analysis = np_proc.analyze_news_with_qwen(text, "Tencent")
    print(f"Analysis Result: {analysis}")
    
    # Test full flow
    emb = np_proc.process_daily_news([text], "Tencent")
    print(f"Final Embedding Shape: {emb.shape}")
