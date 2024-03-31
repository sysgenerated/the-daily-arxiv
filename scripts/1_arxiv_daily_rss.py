#!/usr/bin/env python
# coding: utf-8

# ## 1. Load required libraries

# In[ ]:


import os
import feedparser
import json
import gzip

from feedparser.util import FeedParserDict
from datetime import datetime


# ## 2. Define retrieval functions

# In[ ]:


# arxiv provides an rss feed for all current day articles 
def retrieve_arxiv_rss(section: str = "cs") -> FeedParserDict | None:
    base_url = "http://rss.arxiv.org/rss"
    feed_url = f"{base_url}/{section}"

    try:
        feed = feedparser.parse(feed_url)

        # Check for errors in parsing
        if feed.bozo:
            raise feedparser.bozo_exception(feed.bozo_exception.getMessage())
        
        return feed

    except (ValueError, IOError, feedparser.ParseError) as e:
        print(f"Error fetching feed entries: {e}")
        return None


def parse_arxiv_rss(feed: feedparser.FeedParserDict) -> list[dict[str, any]]:
        data = []

        # Iterate over items and extract information
        for item in feed.entries:
            title = item.title
            link = item.link
            summary = item.summary
            
            # Extract abstract from the summary
            if summary:
                summary_parts = summary.split('\nAbstract:')
                abstract = summary_parts[1].strip() if len(summary_parts) > 1 else ''
            else:
                abstract = ''
            
            guid = item.id
            category = ','.join(tag.term for tag in item.tags)
            rights = item.rights
            author = item.author
            
            # Append extracted information as a dictionary to the data list
            data.append({
                'title': title,
                'abstract': abstract,
                'guid': guid,
                'category': category,
                'link': link,
                'rights': rights,
                'creator': author,
            })

        return data
    

def get_arxiv_rss_name(feed: feedparser.FeedParserDict) -> str:
        published_date = datetime.strptime(feed.feed.published, '%a, %d %b %Y %H:%M:%S %z')
        published_date = published_date.strftime('%Y_%m_%d')
        section = feed.feed.link.split("/")[-1]
        return f"{published_date}_{section}"


def write_json(filename: str, parsed_feed: list[dict[str, any]]) -> None:
    subdirectory = "../data"
    os.makedirs(subdirectory, exist_ok=True)
    filepath = os.path.join(subdirectory, filename)
    print(f"writing json file: {filepath}.json.gz")
    with gzip.open(f"{filepath}.json.gz", "w") as f:
        f.write(json.dumps(parsed_feed).encode("utf-8"))


# valid sections = cs, econ, eess, math, astro-ph, cond-mat, gr-qc, hep-ex, 
#                  hep-lat, hep-ph, hep-th, math-ph, nlin, nucl-ex, 
#                  nucl-th, physics, quant-ph, q-bio, q-fin, stat 
def daily_processing(section: str="cs") -> None:
    feed = retrieve_arxiv_rss(section)
    if feed is not None:
        print("retrieved feed ... ")
        filename = get_arxiv_rss_name(feed)
        parsed_feed = parse_arxiv_rss(feed)
        print("parsed feed ... ")
        write_json(filename, parsed_feed)
        print("wrote json file ... ")


# ## 3. Retrieve Arxiv articles

# In[ ]:


if __name__ == "__main__":
    daily_processing()

