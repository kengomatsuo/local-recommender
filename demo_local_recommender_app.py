
import streamlit as st
import pandas as pd
import random
import secrets
import nacl.signing
import nacl.encoding

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ------------------- ZKP SETUP -------------------

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

# ------------------ Local Recommender -------------------

class LocalRecommender:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
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
        y = df['interest_score']
        preprocessor = ColumnTransformer(transformers=[
            ('topic', CountVectorizer(), 'topic'),
            ('hashtags', CountVectorizer(), 'hashtags_str'),
            ('num', StandardScaler(), ['liked', 'commented', 'duration', 'time_watched'])
        ])
        self.pipeline = Pipeline(steps=[
            ('prep', preprocessor),
            ('model', Ridge(alpha=self.alpha))
        ])
        self.pipeline.fit(X, y)
        self.trained = True

    def recommend(self, df_user: pd.DataFrame, normalize=True):
        if not self.trained:
            raise RuntimeError("Model must be trained using `fit()` before calling `recommend()`.")
        df = df_user.copy()
        df['liked'] = df['liked'].astype(int)
        df['commented'] = df['commented'].astype(int)
        df['hashtags_str'] = df['hashtags'].apply(lambda x: " ".join(x))
        topic_scores = {}
        for topic in self.topics:
            topic_samples = df[df['topic'] == topic]
            if not topic_samples.empty:
                preds = self.pipeline.predict(topic_samples[['topic', 'liked', 'commented', 'duration', 'time_watched', 'hashtags_str']])
                topic_scores[topic] = round(preds.mean(), 3)
        if normalize:
            total = sum(topic_scores.values())
            return {k: round(v / total, 3) for k, v in topic_scores.items()} if total > 0 else topic_scores
        return topic_scores

# ------------------- Streamlit UI -------------------

recommender = LocalRecommender()

if "interactions" not in st.session_state:
    st.session_state.interactions = []

topics = ["tech", "art", "sports", "music", "news", "fashion", "food", "gaming"]
tags_pool = ["ai", "design", "funny", "news", "vlog", "review"]

def generate_post():
    topic = random.choice(topics)
    hashtags = random.sample(tags_pool, k=random.randint(1, 3))
    duration = round(random.uniform(5.0, 60.0), 2)
    return {"topic": topic, "hashtags": hashtags, "duration": duration}

st.title("Local Recommender with ZKP Demo")

if "current_post" not in st.session_state:
    st.session_state.current_post = generate_post()

post = st.session_state.current_post
st.subheader(f"Topic: {post['topic']}")
st.text(f"Hashtags: {' '.join(post['hashtags'])}")
st.text(f"Video Duration: {post['duration']} seconds")

# User input
liked = st.checkbox("Like")
commented = st.checkbox("Comment")
interest_flag = st.radio("Feedback", options=["Neutral", "Interested", "Not Interested"], index=0)
time_watched = st.slider("How long did you watch this post?", 0.0, post['duration'] * 2, post['duration'] / 2)

if st.button("Next"):
    base_score = time_watched / post["duration"]
    score = max(0.0, min(1.0, 
        (0.2 * base_score) +
        (0.3 if liked else 0) +
        (0.3 if commented else 0) +
        (0.2 if interest_flag == "Interested" else -0.2 if interest_flag == "Not Interested" else 0)
    ))

    st.session_state.interactions.append({
        "topic": post["topic"],
        "hashtags": post["hashtags"],
        "liked": liked,
        "commented": commented,
        "time_watched": time_watched,
        "duration": post["duration"],
        "interest_score": round(score, 3)
    })
    st.session_state.current_post = generate_post()

if st.session_state.interactions:
    df = pd.DataFrame(st.session_state.interactions)
    st.subheader("Interaction Log")
    st.dataframe(df)

    if len(df) >= 10:
        st.subheader("Predicted Topic Preferences")
        recommender.fit(df.tail(100))
        weights = recommender.recommend(df)
        st.json(weights)

        st.subheader("ZKP Proof (Client-side)")
        challenge = st.session_state.challenge
        proof = generate_zkp_proof(challenge)
        st.text(f"Challenge: {challenge}")
        st.text(f"Proof: {proof.signature[:10].hex()}...")

        st.subheader("ZKP Verification (Server-side)")
        verified = verify_zkp_proof(proof, challenge, st.session_state.verify_key)
        st.success("Proof Verified ✅" if verified else "Invalid Proof ❌")
