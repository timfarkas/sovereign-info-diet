#!/usr/bin/env python3

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

load_dotenv()

class LLMSummarizer:
    def __init__(self, model_name: str = "gpt-5-mini"):
        """Initialize the LLM summarizer with OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        
    def create_summary_prompt(self, posts: List[Dict]) -> str:
        """Create the summarization prompt for all posts"""
        
        from config import TIME_HORIZON_DAYS, SUMMARY_PROMPT_TEMPLATE
        
        # format all posts into a digest
        posts_content = ""
        for i, post in enumerate(posts, 1):
            posts_content += f"\n{i}. {post['title']} [{post['score']} pts]\n"
            posts_content += f"   r/{post['subreddit']}\n"
            
            if post.get('selftext'):
                preview = post['selftext'][:200] + "..." if len(post['selftext']) > 200 else post['selftext']
                posts_content += f"   Text: {preview}\n"
            
            # include top 3 comments for context
            if post.get('comments'):
                posts_content += "   Top comments:\n"
                for comment in post['comments'][:3]:
                    comment_preview = comment['body'][:150] + "..." if len(comment['body']) > 150 else comment['body']
                    posts_content += f"   - [{comment['score']}pts] {comment_preview}\n"
        
        # format the prompt template
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            TIME_HORIZON_DAYS=TIME_HORIZON_DAYS,
            posts_content=posts_content
        )
        
        return prompt
    
    def summarize_posts(self, posts: List[Dict]) -> str:
        """Generate a summary of all posts"""
        
        print(f"\nAnalyzing {len(posts)} posts with {self.model}...")
        
        prompt = self.create_summary_prompt(posts)
        
        try:
            # GPT-5 has different parameter requirements
            if 'gpt-5' in self.model:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a sharp, insightful AI/tech news analyst with a good sense of humor."},
                        {"role": "user", "content": prompt}
                    ]
                    # no temperature or max_tokens for GPT-5
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a sharp, insightful AI/tech news analyst with a good sense of humor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
            summary = response.choices[0].message.content.strip()
            
            print("Summary generated successfully")
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Failed to generate summary: {str(e)}"
    
    def save_summary(self, summary: str, posts_analyzed: int, filename: str = None):
        """Save the summary to a file"""
        
        from config import TIME_HORIZON_DAYS
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"summary_{timestamp}.md"
        
        output_dir = Path("extracts/summaries")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        filepath = output_dir / filename
        
        # add header
        full_content = f"""# AI Digest - {datetime.now().strftime("%Y-%m-%d")}

*Analyzed {posts_analyzed} posts from the past {TIME_HORIZON_DAYS} days*

---

{summary}

---

*Generated at {datetime.now().strftime("%I:%M %p")}*
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"Summary saved to {filepath}")
        return str(filepath)


if __name__ == "__main__":
    # load scraped posts
    import glob
    latest_scrape = max(glob.glob("extracts/reddit_data_*.json"))
    
    with open(latest_scrape, 'r') as f:
        posts = json.load(f)
    
    print(f"Loaded {len(posts)} posts from {latest_scrape}")
    
    # generate summary
    summarizer = LLMSummarizer()
    summary = summarizer.summarize_posts(posts)
    
    # save and display
    output_file = summarizer.save_summary(summary, len(posts))
    
    print("\n" + "="*60)
    print(summary)
    print("="*60)