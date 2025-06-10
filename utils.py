# utils.py
import random
import pandas as pd

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
