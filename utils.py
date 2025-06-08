# utils.py
import random

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

def generate_post():
    topic = random.choice(topics)
    hashtags = random.sample(tags_pool, k=random.randint(1, 3))
    duration = round(random.uniform(10.0, 60.0), 2)
    caption = generate_caption(topic)
    return {
        "topic": topic,
        "hashtags": hashtags,
        "duration": duration,
        "caption": caption
    }
