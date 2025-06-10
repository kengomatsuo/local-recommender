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

def generate_post_batch(preferences=None, batch_size=20, noise=0.2):
    """
    Generate a batch of posts influenced by topic and hashtag preferences, with some noise.
    preferences: dict with 'topic_weights' and 'hashtag_weights' (both dicts)
    noise: float in [0,1], higher means more random
    """
    global _posts_cache
    if _posts_cache is None:
        _posts_cache = load_posts()
    posts = _posts_cache
    if not preferences:
        return random.sample(posts, min(batch_size, len(posts)))

    topic_weights = preferences.get('topic_weights', {})
    hashtag_weights = preferences.get('hashtag_weights', {})

    # Compute a score for each post
    scored_posts = []
    for post in posts:
        topic_score = topic_weights.get(post['topic'], 0)
        hashtag_score = sum(hashtag_weights.get(h, 0) for h in post['hashtags']) / (len(post['hashtags']) or 1)
        score = (topic_score + hashtag_score) / 2
        # Add noise
        score = (1 - noise) * score + noise * random.random()
        scored_posts.append((score, post))
    # Sort by score descending, but add randomness
    scored_posts.sort(key=lambda x: x[0], reverse=True)
    # Select top N, but shuffle a bit for noise
    top_posts = [p for _, p in scored_posts[:batch_size*2]]
    random.shuffle(top_posts)
    return top_posts[:batch_size]
