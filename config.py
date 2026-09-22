#!/usr/bin/env python3

# Configuration for the digital info filter

TIME_HORIZON_DAYS = 1  # look back this many days
POSTS_TO_ANALYZE = 30  # number of posts to scrape for analysis

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

**Output format (strict) — respond with a raw HTML fragment, NOT markdown.**
Do not wrap the output in a code fence (no ```html) and do not include <html>/<head>/<body> tags — just the fragment below.
Use only these tags: <h3> for section titles, <h4> for "Further Reading" subtitles, <ul>/<li> for bullets, <strong> for emphasis, <em> for asides/quotes, <a href="URL">text</a> for links. Do not use markdown syntax (no **, no leading -).

<h3>Major Developments</h3>
<ul>
<li>[Each item contains only details taken directly from the post.]</li>
</ul>

<h4>Further Reading — Major Developments</h4>
<ul>
<li><a href="URL">[Actual URL if found in posts]</a></li>
</ul>
[Omit this Further Reading block entirely if no links were found.]

<h3>Sentiment Shifts</h3>
<ul>
<li>[Same style, with quotes where possible.]</li>
</ul>

<h4>Further Reading — Sentiment Shifts</h4>
<ul>
<li><a href="URL">[Actual URL if found in posts]</a></li>
</ul>
[If none found, write: <li>No external links found in posts</li>]

<h3>The Absurd Corner</h3>
<ul>
<li>[Short, witty descriptions tied to what's in the source.]</li>
</ul>

<h4>Further Reading — Absurd Corner</h4>
<ul>
<li><a href="URL">[Actual URL if found in posts]</a></li>
</ul>
[If none found, write: <li>No external links found in posts</li>]

Posts from the past {TIME_HORIZON_DAYS} days:
{posts_content}
"""

SUBREDDITS = ["singularity", "DeepLearning", "MachineLearning", "LocalLLaMA"]  # multiple subreddits for better coverage

SORT_BY = "hot"  # "hot", "new", or "top" - top gets best posts from time period
