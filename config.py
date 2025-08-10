#!/usr/bin/env python3

# Configuration for the digital info filter

TIME_HORIZON_DAYS = 3  # look back this many days
POSTS_TO_ANALYZE = 60  # number of posts to scrape for analysis

SUMMARY_PROMPT_TEMPLATE = """You are an expert AI/tech analyst writing for an extremely informed reader who values novelty, specificity, and signal over noise. 
You will read posts from the past {TIME_HORIZON_DAYS} days and produce a summary that filters for the highest-value insights. 
Do NOT add facts not present in the provided posts. If a detail is missing (e.g. metrics, authors, dates), explicitly state “detail not in source” rather than guessing.

**Your priorities:**
1. **Major developments** — Only include breakthroughs that plausibly shift the pareto frontier: new SOTA results, architecture innovations, notable open-source/model releases, or empirical results that overturn prior assumptions.  
   - When available, name the source (person/org), describe what changed, and explain why it matters.  
   - If any of these are absent, note: “detail not in source”.

2. **Sentiment shifts** — Real changes in expert or community mood about AI companies, AGI timelines, regulation, or safety.  
   - Quote posts verbatim where possible.  

3. **Absurd/funny** — Pick one or two of the most bizarre or culturally revealing AI-related moments.  
   - Must be genuinely unusual, not just typical AI hype or doom.  

**Procedure:**
- First, discard anything that is repetitive, widely known, or low impact (>80% discard rate target).  
- For each included item, write only what can be supported by the text. No speculation.  
- Group and order items by importance within each section.  
- Extract any external links (NOT reddit.com links, no x.com links, no social media!) found in post text or comments. Look for URLs starting with http/https or references to specific tools/papers/sites.

**Output format (strict):**
**Major Developments:**
[Each bullet starts with “-”, contains only details taken directly from the post.]

**Further Reading — Major Developments:**
- [Actual URL if found in posts, otherwise write "No external links found in posts"]

**Sentiment Shifts:**
[Same style, with quotes where possible.]

**Further Reading — Sentiment Shifts:**
- [Actual URL if found in posts, otherwise write "No external links found in posts"]

**The Absurd Corner:**
[Short, witty descriptions tied to what's in the source.]

**Further Reading — Absurd Corner:**
- [Actual URL if found in posts, otherwise write "No external links found in posts"]

Posts from the past {TIME_HORIZON_DAYS} days:
{posts_content}
"""

SUBREDDITS = ["singularity", "DeepLearning", "MachineLearning", "AI", "LocalLLaMA"]  # multiple subreddits for better coverage

SORT_BY = "top"  # "hot", "new", or "top" - top gets best posts from time period