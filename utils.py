# utils.py
import random
import pandas as pd

topics = ["tech", "art", "sports", "music", "news", "fashion", "food", "gaming"]
tags_pool = ["ai", "design", "funny", "news", "vlog", "review"]

def generate_caption(topic):
    templates = {
        "tech": "Understanding the future of {}.",
        "art": "The evolution of modern {}.",
        "sports": "Why {} is more exciting than ever.",
        "music": "How {} shaped music history.",
        "news": "Breaking news: What you need to know about {}.",
        "fashion": "This season's top trends in {}.",
        "food": "How to cook the perfect {} dish.",
        "gaming": "Why everyone is talking about {}."
    }
    return templates.get(topic, "An interesting look at {}.").format(topic)

def load_posts():
    df = pd.read_csv("posts.csv")
    posts = df.to_dict(orient="records")
    for post in posts:
        # Convert hashtags from comma-separated string to list
        if isinstance(post["hashtags"], str):
            post["hashtags"] = [h.strip() for h in post["hashtags"].split(",")]
    return posts

# Replace generate_post with a function that samples from posts.csv
_posts_cache = None
def generate_post():
    global _posts_cache
    if _posts_cache is None:
        _posts_cache = load_posts()
    return random.choice(_posts_cache)
