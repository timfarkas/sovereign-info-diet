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
    def __init__(self, base_dir: str = "extracts"):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_SECRET"),
            user_agent="tim-filter/0.1 by timfarkas"
        )
        self.reddit.read_only = True
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.images_dir = self.base_dir / "images"
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
                       max_replies: int = 3, condensed: bool = True) -> Dict[str, Any]:
        """Recursively process comment and its replies"""
        
        # process replies first to check if we have any
        replies = []
        if depth < max_depth and hasattr(comment, 'replies'):
            for reply in comment.replies[:max_replies]:
                if isinstance(reply, praw.models.Comment):
                    replies.append(
                        self.process_comment(reply, depth + 1, max_depth, max_replies, condensed)
                    )
        
        if condensed:
            # minimal format
            comment_data = {
                'body': comment.body,
                'score': comment.score,
            }
            # only add replies if non-empty
            if replies:
                comment_data['replies'] = replies
        else:
            # full format
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
                'replies': replies
            }
            
            # download embedded images
            downloaded_images = []
            for i, img_url in enumerate(embedded_images):
                img_path = self.download_image(img_url, f"comment_{comment.id}", i)
                if img_path:
                    downloaded_images.append(img_path)
            comment_data['downloaded_images'] = downloaded_images
        
        return comment_data
    
    def process_post(self, submission, max_comments: int = 5, 
                    max_comment_depth: int = 1, max_replies: int = 3, 
                    condensed: bool = True) -> Dict[str, Any]:
        """Process a Reddit post and return structured data"""
        
        # always process images since they're important
        images = []
        
        # handle different post types
        if 'i.redd.it' in submission.url:
            # direct image
            img_path = self.download_image(submission.url, submission.id)
            if img_path:
                if condensed:
                    images.append(img_path)
                else:
                    images.append({
                        'url': submission.url,
                        'local_path': img_path,
                        'type': 'main'
                    })
        
        elif hasattr(submission, 'is_gallery') and submission.is_gallery:
            # gallery post
            for j, item_id in enumerate(submission.media_metadata):
                media = submission.media_metadata[item_id]
                # gallery items that are gifs/videos carry an 's' preview dict too,
                # but without a 'u' key -- only static images have one
                if 's' in media and 'u' in media['s']:
                    img_url = media['s']['u'].replace('&amp;', '&')
                    img_url = img_url.replace('preview.redd.it', 'i.redd.it')
                    img_url = img_url.split('?')[0]
                    
                    img_path = self.download_image(img_url, submission.id, j)
                    if img_path:
                        if condensed:
                            images.append(img_path)
                        else:
                            images.append({
                                'url': img_url,
                                'local_path': img_path,
                                'type': 'gallery',
                                'index': j
                            })
        
        if condensed:
            # minimal format
            post_data = {
                'title': submission.title,
                'score': submission.score,
                'subreddit': submission.subreddit.display_name,
                'selftext': submission.selftext if submission.selftext else None,
                # external destination of a link post -- without this a link post
                # contributes only its title, and the article it points at is lost
                'url': None if submission.is_self else submission.url,
                'images': images if images else None,
                'comments': []
            }
            # remove None values
            post_data = {k: v for k, v in post_data.items() if v is not None}
        else:
            # full format
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
                'images': images,
                'comments': []
            }
        
        # process comments
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:max_comments]:
            if isinstance(comment, praw.models.Comment):
                post_data['comments'].append(
                    self.process_comment(comment, depth=0, max_depth=max_comment_depth, 
                                       max_replies=max_replies, condensed=condensed)
                )
        
        return post_data
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 10, 
                        sort: str = 'hot', condensed: bool = True,
                        time_horizon_days: int = None) -> List[Dict[str, Any]]:
        """Scrape posts from a subreddit, optionally filtering by time"""
        
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # get posts based on sort method
        if sort == 'hot':
            posts = subreddit.hot(limit=limit * 3)  # fetch extra to filter
        elif sort == 'new':
            posts = subreddit.new(limit=limit * 3)
        elif sort == 'top':
            posts = subreddit.top(limit=limit * 3, time_filter='week')
        else:
            posts = subreddit.new(limit=limit * 3)
        
        # filter by time if specified
        if time_horizon_days:
            from datetime import datetime, timedelta, timezone
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=time_horizon_days)
            cutoff_timestamp = cutoff_time.timestamp()
        
        results = []
        for submission in posts:
            # skip stickied posts
            if submission.stickied:
                continue
            
            # filter by time if specified
            if time_horizon_days and submission.created_utc < cutoff_timestamp:
                continue
            
            if len(results) >= limit:
                break
            
            print(f"Processing: {submission.title[:60]}...")
            post_data = self.process_post(submission, condensed=condensed)
            results.append(post_data)
        
        print(f"Found {len(results)} posts within time horizon")
        return results
    
    def save_to_json(self, data: List[Dict], filename: str = None):
        """Save scraped data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_data_{timestamp}.json"
        
        filepath = self.base_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {filepath}")
        return str(filepath)


if __name__ == "__main__":
    from config import POSTS_TO_ANALYZE, SUBREDDITS, SORT_BY, TIME_HORIZON_DAYS
    
    scraper = RedditScraper()
    
    all_posts = []
    failed_subreddits = []
    posts_per_sub = POSTS_TO_ANALYZE // len(SUBREDDITS)  # divide quota among subreddits

    # scrape each subreddit
    for subreddit in SUBREDDITS:
        print(f"\n📊 Scraping r/{subreddit}...")
        try:
            posts = scraper.scrape_subreddit(
                subreddit,
                limit=posts_per_sub,
                sort=SORT_BY,
                condensed=True,
                time_horizon_days=TIME_HORIZON_DAYS
            )
        except Exception as e:
            # one bad subreddit shouldn't cost the posts already scraped from the others
            print(f"[reddit_scraper] r/{subreddit} failed: {e} -- skipping, keeping what's scraped so far")
            failed_subreddits.append(subreddit)
            continue
        all_posts.extend(posts)
    
    # save to json
    output_file = scraper.save_to_json(all_posts)
    
    # print summary
    print(f"\nTotal scraped: {len(all_posts)} posts across {len(SUBREDDITS)} subreddits")
    total_comments = sum(len(p['comments']) for p in all_posts)
    total_images = sum(len(p.get('images', [])) for p in all_posts)
    total_replies = sum(sum(len(c.get('replies', [])) for c in p['comments']) for p in all_posts)
    print(f"Total top-level comments: {total_comments}")
    print(f"Total replies: {total_replies}")
    print(f"Total images downloaded: {total_images}")
    
    # show breakdown by subreddit
    from collections import Counter
    sub_counts = Counter(p['subreddit'] for p in all_posts)
    print(f"\nPosts by subreddit:")
    for sub, count in sub_counts.items():
        print(f"  r/{sub}: {count}")

    if failed_subreddits:
        import sys
        print(f"\n[reddit_scraper] failed subreddits: {', '.join(failed_subreddits)}")
        sys.exit(1)