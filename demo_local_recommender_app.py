
import streamlit as st
import pandas as pd
import random
import joblib

# Load class
LocalRecommender = joblib.load("LocalRecommenderClass.pkl")
recommender = LocalRecommender()

# Session state for interactions
if "interactions" not in st.session_state:
    st.session_state.interactions = []

# Topics and tags pool
topics = ["tech", "art", "sports", "music", "news", "fashion", "food", "gaming"]
tags_pool = ["ai", "design", "funny", "news", "vlog", "review"]

# Simulate a post
def generate_post():
    topic = random.choice(topics)
    hashtags = random.sample(tags_pool, k=random.randint(1, 3))
    duration = round(random.uniform(5.0, 60.0), 2)
    return {"topic": topic, "hashtags": hashtags, "duration": duration}

st.title("Local Recommender Demo")

# Show a simulated post
if "current_post" not in st.session_state:
    st.session_state.current_post = generate_post()

post = st.session_state.current_post
st.subheader(f"Topic: {post['topic']}")
st.text(f"Hashtags: {' '.join(post['hashtags'])}")
st.text(f"Video Duration: {post['duration']} seconds")

# User interaction
liked = st.button("Like")
commented = st.button("Comment")
interested = st.button("Mark as Interested")
not_interested = st.button("Mark as Not Interested")
time_watched = st.slider("How long did you watch this post?", 0.0, post['duration'] * 2, post['duration'] / 2)

# Save interaction
if liked or commented or interested or not_interested:
    score = 0.7 if interested else (0.3 if liked or commented else 0.0)
    st.session_state.interactions.append({
        "topic": post["topic"],
        "hashtags": post["hashtags"],
        "liked": liked,
        "commented": commented,
        "time_watched": time_watched,
        "duration": post["duration"],
        "interest_score": score
    })
    st.session_state.current_post = generate_post()
    st.experimental_rerun()

# Show last interactions
if st.session_state.interactions:
    df = pd.DataFrame(st.session_state.interactions)
    st.subheader("Interaction Log")
    st.dataframe(df)

    if len(df) >= 10:
        st.subheader("Predicted Topic Preferences")
        recommender.fit(df.tail(100))
        weights = recommender.recommend(df)
        weight_df = pd.DataFrame(weights.items(), columns=["Topic", "Weight"])
        st.bar_chart(weight_df.set_index("Topic"))
