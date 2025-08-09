#!/usr/bin/env python3

import os
import praw
from dotenv import load_dotenv
import requests
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Optional, Any
import hashlib

load_dotenv()

class RedditScraper:
    def __init__(self, images_dir: str = "images"):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent="tim-filter/0.1 by timfarkas"
        )
        self.reddit.read_only = True
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
    
    def download_image(self, url: str, post_id: str, img_index: int = 0) -> Optional[str]:
        """Download image and return relative path"""
        try:
            if any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                response = requests.get(url, headers={'User-Agent': 'tim-filter/0.1'})
                if response.status_code == 200:
                    ext = url.split('.')[-1].split('?')[0]
                    filename = self.images_dir / f"{post_id}_{img_index}.{ext}"
                    
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    return str(filename)
        except Exception as e:
            print(f"Image download failed: {e}")
        return None
    
    def extract_comment_images(self, comment_body: str) -> List[str]:
        """Extract image URLs from comment markdown"""
        images = []
        # look for markdown image syntax ![alt](url)
        import re
        pattern = r'!\[.*?\]\((.*?)\)'
        matches = re.findall(pattern, comment_body)
        
        for url in matches:
            # handle reddit preview urls
            if 'preview.redd.it' in url:
                url = url.replace('preview.redd.it', 'i.redd.it')
                url = url.split('?')[0]
            images.append(url)
        
        return images
    
    def process_comment(self, comment, depth: int = 0, max_depth: int = 1, 
                       max_replies: int = 3) -> Dict[str, Any]:
        """Recursively process comment and its replies"""
        
        # extract any embedded images in comment
        embedded_images = self.extract_comment_images(comment.body)
        
        comment_data = {
            'id': comment.id,
            'author': str(comment.author) if comment.author else '[deleted]',
            'body': comment.body,
            'score': comment.score,
            'created_utc': comment.created_utc,
            'created_human': datetime.fromtimestamp(comment.created_utc).isoformat(),
            'depth': depth,
            'permalink': f"https://reddit.com{comment.permalink}",
            'embedded_images': embedded_images,
            'replies': []
        }
        
        # download embedded images
        downloaded_images = []
        for i, img_url in enumerate(embedded_images):
            img_path = self.download_image(img_url, f"comment_{comment.id}", i)
            if img_path:
                downloaded_images.append(img_path)
        comment_data['downloaded_images'] = downloaded_images
        
        # process replies recursively
        if depth < max_depth and hasattr(comment, 'replies'):
            for reply in comment.replies[:max_replies]:
                if isinstance(reply, praw.models.Comment):
                    comment_data['replies'].append(
                        self.process_comment(reply, depth + 1, max_depth, max_replies)
                    )
        
        return comment_data
    
    def process_post(self, submission, max_comments: int = 5, 
                    max_comment_depth: int = 1, max_replies: int = 3) -> Dict[str, Any]:
        """Process a Reddit post and return structured data"""
        
        post_data = {
            'id': submission.id,
            'title': submission.title,
            'author': str(submission.author) if submission.author else '[deleted]',
            'selftext': submission.selftext,
            'url': submission.url,
            'permalink': f"https://reddit.com{submission.permalink}",
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc,
            'created_human': datetime.fromtimestamp(submission.created_utc).isoformat(),
            'subreddit': submission.subreddit.display_name,
            'is_video': submission.is_video,
            'is_gallery': hasattr(submission, 'is_gallery') and submission.is_gallery,
            'images': [],
            'comments': []
        }
        
        # handle different post types
        if 'i.redd.it' in submission.url:
            # direct image
            img_path = self.download_image(submission.url, submission.id)
            if img_path:
                post_data['images'].append({
                    'url': submission.url,
                    'local_path': img_path,
                    'type': 'main'
                })
        
        elif hasattr(submission, 'is_gallery') and submission.is_gallery:
            # gallery post
            for j, item_id in enumerate(submission.media_metadata):
                media = submission.media_metadata[item_id]
                if 's' in media:
                    img_url = media['s']['u'].replace('&amp;', '&')
                    img_url = img_url.replace('preview.redd.it', 'i.redd.it')
                    img_url = img_url.split('?')[0]
                    
                    img_path = self.download_image(img_url, submission.id, j)
                    if img_path:
                        post_data['images'].append({
                            'url': img_url,
                            'local_path': img_path,
                            'type': 'gallery',
                            'index': j
                        })
        
        # process comments
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:max_comments]:
            if isinstance(comment, praw.models.Comment):
                post_data['comments'].append(
                    self.process_comment(comment, depth=0, max_depth=max_comment_depth, 
                                       max_replies=max_replies)
                )
        
        return post_data
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 10, 
                        sort: str = 'hot') -> List[Dict[str, Any]]:
        """Scrape posts from a subreddit"""
        
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # get posts based on sort method
        if sort == 'hot':
            posts = subreddit.hot(limit=limit + 1)  # +1 for potential sticky
        elif sort == 'new':
            posts = subreddit.new(limit=limit)
        elif sort == 'top':
            posts = subreddit.top(limit=limit, time_filter='day')
        else:
            posts = subreddit.hot(limit=limit + 1)
        
        results = []
        for i, submission in enumerate(posts):
            # skip stickied posts
            if i == 0 and submission.stickied:
                continue
            
            if len(results) >= limit:
                break
            
            print(f"Processing: {submission.title[:60]}...")
            post_data = self.process_post(submission)
            results.append(post_data)
        
        return results
    
    def save_to_json(self, data: List[Dict], filename: str = None):
        """Save scraped data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_data_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {filename}")
        return filename


if __name__ == "__main__":
    scraper = RedditScraper()
    
    # scrape r/singularity with configured limits
    # 10 posts, top 5 comments each, 1 level deep with 3 replies max
    posts = scraper.scrape_subreddit("singularity", limit=10, sort='hot')
    
    # save to json
    output_file = scraper.save_to_json(posts)
    
    # print summary
    print(f"\nScraped {len(posts)} posts")
    total_comments = sum(len(p['comments']) for p in posts)
    total_images = sum(len(p['images']) for p in posts)
    total_replies = sum(sum(len(c['replies']) for c in p['comments']) for p in posts)
    print(f"Total top-level comments: {total_comments}")
    print(f"Total replies: {total_replies}")
    print(f"Total images downloaded: {total_images}")
    
    # show sample of structure
    if posts:
        print(f"\nSample post structure:")
        print(f"- Title: {posts[0]['title'][:60]}...")
        print(f"- Score: {posts[0]['score']}")
        print(f"- Comments: {len(posts[0]['comments'])}")
        if posts[0]['comments']:
            print(f"- First comment has {len(posts[0]['comments'][0]['replies'])} replies")