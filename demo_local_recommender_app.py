import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import tracemalloc
from keybert import KeyBERT
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ------------------- Recommender Classifier -------------------
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
    return {"topic": topic, "hashtags": hashtags, "duration": duration, "caption": caption}

class KeyBERTVectorizer:
    def __init__(self, model='all-MiniLM-L6-v2', n_keywords=5):
        self.kw_model = KeyBERT(model=model)
        self.n_keywords = n_keywords

    def transform(self, texts):
        keywords_list = []
        for text in texts:
            keywords = self.kw_model.extract_keywords(text, top_n=self.n_keywords)
            keywords_list.append(" ".join([kw[0] for kw in keywords]))
        return keywords_list

    def fit(self, texts, y=None):
        return self

class LocalRecommenderClassifier:
    def __init__(self):
        self.pipeline = None
        self.trained = False

    def fit(self, df_user: pd.DataFrame):
        df = df_user[df_user["engaged"] != 1].copy()
        df["liked"] = df["liked"].astype(int)
        df["commented"] = df["commented"].astype(int)
        df["hashtags_str"] = df["hashtags"].apply(lambda x: " ".join(x))
        
        X = df[
            ["topic", "liked", "commented", "duration", "time_watched", "hashtags_str", "caption"]
        ]
        y = df["engaged"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("topic", CountVectorizer(), "topic"),
                ("hashtags", CountVectorizer(), "hashtags_str"),
                ("tfidf", TfidfVectorizer(stop_words='english', max_features=50), "caption"),
                ("keybert", Pipeline([
                    ('kw', KeyBERTVectorizer(n_keywords=5)),
                    ('vec', CountVectorizer())
                ]), "caption"),
                (
                    "num",
                    StandardScaler(),
                    ["liked", "commented", "duration", "time_watched"],
                ),
            ]
        )

        self.pipeline = Pipeline(
            [
                ("prep", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        self.pipeline.fit(X, y)
        self.trained = True

    def recommend(self, df_user: pd.DataFrame, normalize=True):
        if not self.trained:
            return {}, {}
        df = df_user.copy()
        df["liked"] = df["liked"].astype(int)
        df["commented"] = df["commented"].astype(int)
        df["hashtags_str"] = df["hashtags"].apply(lambda x: " ".join(x))

        topic_scores = {}
        hashtag_scores = {}

        for topic in df["topic"].unique():
            samples = df[df["topic"] == topic]
            if not samples.empty:
                X_topic = samples[
                    [
                        "topic",
                        "liked",
                        "commented",
                        "duration",
                        "time_watched",
                        "hashtags_str",
                        "caption",
                    ]
                ]
                probas = self.pipeline.predict_proba(X_topic)
                if probas.shape[1] > 1:
                    preds = probas[:, 1]
                else:
                    preds = np.zeros(len(probas))
                topic_scores[topic] = round(preds.mean(), 3)

        all_hashtags = set(tag for tags in df["hashtags"] for tag in tags)
        for tag in all_hashtags:
            samples = df[df["hashtags"].apply(lambda tags: tag in tags)]
            if not samples.empty:
                X_tag = samples[
                    [
                        "topic",
                        "liked",
                        "commented",
                        "duration",
                        "time_watched",
                        "hashtags_str",
                        "caption",
                    ]
                ]
                probas = self.pipeline.predict_proba(X_tag)
                if probas.shape[1] > 1:
                    preds = probas[:, 1]
                else:
                    preds = np.zeros(len(probas))
                hashtag_scores[tag] = round(preds.mean(), 3)

        if normalize:
            t_total = sum(topic_scores.values())
            h_total = sum(hashtag_scores.values())
            topic_scores = (
                {k: round(v / t_total, 3) for k, v in topic_scores.items()}
                if t_total > 0
                else {}
            )
            hashtag_scores = (
                {k: round(v / h_total, 3) for k, v in hashtag_scores.items()}
                if h_total > 0
                else {}
            )

        return topic_scores, hashtag_scores


# ------------------- Streamlit App -------------------
st.title("Local Recommender with Hashtags and Performance Stats")

# Initialize session state variables
if "interactions" not in st.session_state:
    st.session_state.interactions = []

if "current_post" not in st.session_state:
    st.session_state.current_post = generate_post()

# Add a post counter to use in widget keys to force them to reset
if "post_counter" not in st.session_state:
    st.session_state.post_counter = 0


# Add callback for the Next button
def handle_next_click():
    post = st.session_state.current_post
    liked = st.session_state.liked_input
    commented = st.session_state.commented_input
    time_watched = st.session_state[f"watch_input_{st.session_state.post_counter}"]
    interest_flag = st.session_state.interest_input

    if interest_flag == "Interested":
        engaged = 2
    elif interest_flag == "Not Interested" or (
        (time_watched / post["duration"]) < 0.25 and not liked and not commented
    ):
        engaged = 0
    else:
        engaged = int((time_watched / post["duration"]) > 0.8 or liked or commented) + 1

    st.session_state.interactions.append(
        {
            "topic": post["topic"],
            "hashtags": post["hashtags"],
            "liked": liked,
            "commented": commented,
            "time_watched": time_watched,
            "duration": post["duration"],
            "caption": post["caption"],
            "engaged": engaged,
        }
    )
    # Generate new post and increment counter
    st.session_state.current_post = generate_post()
    st.session_state.post_counter += 1

    # Reset input values for next post
    st.session_state.liked_input = False
    st.session_state.commented_input = False
    st.session_state.interest_input = "Neutral"


post = st.session_state.current_post
post_key = str(st.session_state.post_counter)  # Use counter in keys to force refresh

st.subheader(f"Topic: {post['topic']}")
st.text(f"Hashtags: {' '.join(post['hashtags'])}")
st.text(f"Caption: {post['caption']}")
st.text(f"Video Duration: {post['duration']} seconds")

# Add unique keys to widgets based on post_counter
liked = st.checkbox("Liked", key=f"liked_input")
commented = st.checkbox("Commented", key=f"commented_input")
interest_flag = st.radio(
    "Interest Feedback",
    options=["Neutral", "Interested", "Not Interested"],
    index=0,
    key=f"interest_input",
)
time_watched = st.slider(
    "Time Watched", 0.0, post["duration"] * 2, post["duration"] / 2, key=f"watch_input_{st.session_state.post_counter}"
)

# Use on_click instead of if st.button()
st.button("Next", on_click=handle_next_click)

if st.session_state.interactions:
    df = pd.DataFrame(st.session_state.interactions)
    st.subheader("User Interactions")
    st.dataframe(df)

    if len(df[df["engaged"] != 1]) >= 10:
        model_start = time.perf_counter()
        tracemalloc.start()

        model = LocalRecommenderClassifier()
        model.fit(df.tail(100))
        topic_weights, hashtag_weights = model.recommend(df)

        current, peak = tracemalloc.get_traced_memory()
        model_time = time.perf_counter() - model_start
        tracemalloc.stop()

        st.subheader("Inferred Topic Preferences")
        if topic_weights:
            st.bar_chart(
                pd.DataFrame(
                    topic_weights.items(), columns=["Topic", "Weight"]
                ).set_index("Topic")
            )
        else:
            st.info("Topic preferences will appear once the model has enough data.")

        st.subheader("Inferred Hashtag Preferences")
        if hashtag_weights:
            st.bar_chart(
                pd.DataFrame(
                    hashtag_weights.items(), columns=["Hashtag", "Weight"]
                ).set_index("Hashtag")
            )
        else:
            st.info("Hashtag preferences will appear once the model has enough data.")

        st.subheader("Performance Statistics")
        st.markdown(f"**Model Inference Time:** {model_time:.4f} seconds")
        st.markdown(
            f"**Model Memory Usage:** {current / 1024:.2f} KB (current), {peak / 1024:.2f} KB (peak)"
        )
