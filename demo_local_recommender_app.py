
import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Simulate a post
topics = ["tech", "art", "sports", "music", "news", "fashion", "food", "gaming"]
tags_pool = ["ai", "design", "funny", "news", "vlog", "review"]

def generate_post():
    topic = random.choice(topics)
    hashtags = random.sample(tags_pool, k=random.randint(1, 3))
    duration = round(random.uniform(10.0, 60.0), 2)
    return {"topic": topic, "hashtags": hashtags, "duration": duration}

# Classifier-based recommender
class LocalRecommenderClassifier:
    def __init__(self):
        self.pipeline = None
        self.trained = False
        self.topics = []

    def fit(self, df_user: pd.DataFrame):
        df = df_user.copy()
        df['liked'] = df['liked'].astype(int)
        df['commented'] = df['commented'].astype(int)
        df['hashtags_str'] = df['hashtags'].apply(lambda x: " ".join(x))
        self.topics = df['topic'].unique()
        X = df[['topic', 'liked', 'commented', 'duration', 'time_watched', 'hashtags_str']]
        y = df['engaged']

        preprocessor = ColumnTransformer(transformers=[
            ('topic', CountVectorizer(), 'topic'),
            ('hashtags', CountVectorizer(), 'hashtags_str'),
            ('num', StandardScaler(), ['liked', 'commented', 'duration', 'time_watched'])
        ])

        self.pipeline = Pipeline(steps=[
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.pipeline.fit(X, y)
        self.trained = True

    def recommend(self, df_user: pd.DataFrame, normalize=True):
        if not self.trained:
            return {}
        df = df_user.copy()
        df['liked'] = df['liked'].astype(int)
        df['commented'] = df['commented'].astype(int)
        df['hashtags_str'] = df['hashtags'].apply(lambda x: " ".join(x))

        topic_scores = {}
        for topic in self.topics:
            samples = df[df['topic'] == topic]
            if not samples.empty:
                X_topic = samples[['topic', 'liked', 'commented', 'duration', 'time_watched', 'hashtags_str']]
                preds = self.pipeline.predict_proba(X_topic)[:, 1]
                topic_scores[topic] = round(preds.mean(), 3)

        if normalize:
            total = sum(topic_scores.values())
            return {k: round(v / total, 3) for k, v in topic_scores.items()} if total > 0 else topic_scores
        return topic_scores

# Streamlit UI
st.title("Classifier-based Local Recommender with Flags")

if "interactions" not in st.session_state:
    st.session_state.interactions = []

if "current_post" not in st.session_state:
    st.session_state.current_post = generate_post()

post = st.session_state.current_post
st.subheader(f"Topic: {post['topic']}")
st.text(f"Hashtags: {' '.join(post['hashtags'])}")
st.text(f"Video Duration: {post['duration']} seconds")

liked = st.checkbox("Liked", key="liked_input")
commented = st.checkbox("Commented", key="commented_input")
interested = st.checkbox("Interested", key="interested_input")
not_interested = st.checkbox("Not Interested", key="not_interested_input")
time_watched = st.slider("Time Watched", 0.0, post['duration'] * 2, post['duration'] / 2, key="watch_input")

if st.button("Next"):
    engaged = int((time_watched / post['duration']) > 0.8 or liked or commented)
    if interested:
        engaged = 1
    if not_interested:
        engaged = 0

    st.session_state.interactions.append({
        "topic": post["topic"],
        "hashtags": post["hashtags"],
        "liked": liked,
        "commented": commented,
        "time_watched": time_watched,
        "duration": post["duration"],
        "engaged": engaged
    })

    st.session_state.current_post = generate_post()

# Always show interaction log
if st.session_state.interactions:
    df = pd.DataFrame(st.session_state.interactions)
    st.subheader("User Interactions")
    st.dataframe(df)

    # Show predicted weights if enough data
    if len(df) >= 10:
        model = LocalRecommenderClassifier()
        model.fit(df.tail(100))
        weights = model.recommend(df)
        st.subheader("Inferred Topic Preferences")
        st.bar_chart(pd.DataFrame(weights.items(), columns=["Topic", "Weight"]).set_index("Topic"))
