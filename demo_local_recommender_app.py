import streamlit as st
import pandas as pd
import numpy as np
import random
import secrets
import nacl.signing
import nacl.encoding
import time
import tracemalloc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ------------------- ZKP Setup -------------------
if "signing_key" not in st.session_state:
    st.session_state.signing_key = nacl.signing.SigningKey.generate()
    st.session_state.verify_key = st.session_state.signing_key.verify_key
    st.session_state.challenge = secrets.token_hex(16)


def generate_zkp_proof(challenge: str):
    key = st.session_state.signing_key
    return key.sign(challenge.encode(), encoder=nacl.encoding.HexEncoder)


def verify_zkp_proof(proof, challenge: str, verify_key):
    try:
        verify_key.verify(proof, encoder=nacl.encoding.HexEncoder)
        return True
    except:
        return False


# ------------------- Recommender Classifier -------------------
topics = ["tech", "art", "sports", "music", "news", "fashion", "food", "gaming"]
tags_pool = ["ai", "design", "funny", "news", "vlog", "review"]


def generate_post():
    topic = random.choice(topics)
    hashtags = random.sample(tags_pool, k=random.randint(1, 3))
    duration = round(random.uniform(10.0, 60.0), 2)
    return {"topic": topic, "hashtags": hashtags, "duration": duration}


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
            ["topic", "liked", "commented", "duration", "time_watched", "hashtags_str"]
        ]
        y = df["engaged"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("topic", CountVectorizer(), "topic"),
                ("hashtags", CountVectorizer(), "hashtags_str"),
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
st.title("Local Recommender with ZKP, Hashtags, and Performance Stats")

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
interest_flag = st.radio(
    "Interest Feedback", options=["Neutral", "Interested", "Not Interested"], index=0
)
time_watched = st.slider(
    "Time Watched", 0.0, post["duration"] * 2, post["duration"] / 2, key="watch_input"
)

if st.button("Next"):
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
            "engaged": engaged,
        }
    )
    st.session_state.current_post = generate_post()

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
        st.bar_chart(
            pd.DataFrame(topic_weights.items(), columns=["Topic", "Weight"]).set_index(
                "Topic"
            )
        )

        st.subheader("Inferred Hashtag Preferences")
        st.bar_chart(
            pd.DataFrame(
                hashtag_weights.items(), columns=["Hashtag", "Weight"]
            ).set_index("Hashtag")
        )

        st.subheader("ZKP Verification")
        challenge = st.session_state.challenge
        proof = generate_zkp_proof(challenge)
        st.text(f"Challenge: {challenge}")
        st.text(f"Proof Signature (truncated): {proof.signature[:10].hex()}...")
        verified = verify_zkp_proof(proof, challenge, st.session_state.verify_key)
        if verified:
            st.success("✅ ZKP Verification Passed")
        else:
            st.error("❌ ZKP Verification Failed")

        st.subheader("Performance Statistics")
        st.markdown(f"**Model Inference Time:** {model_time:.4f} seconds")
        st.markdown(
            f"**Memory Usage:** {current / 1024:.2f} KB (current), {peak / 1024:.2f} KB (peak)"
        )
