
import streamlit as st
import pandas as pd
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# LocalRecommender class definition
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

# Initialize recommender
recommender = LocalRecommender()

# Initialize session state
if "interactions" not in st.session_state:
    st.session_state.interactions = []

topics = ["tech", "art", "sports", "music", "news", "fashion", "food", "gaming"]
tags_pool = ["ai", "design", "funny", "news", "vlog", "review"]

# Generate a random post
def generate_post():
    topic = random.choice(topics)
    hashtags = random.sample(tags_pool, k=random.randint(1, 3))
    duration = round(random.uniform(5.0, 60.0), 2)
    return {"topic": topic, "hashtags": hashtags, "duration": duration}

# Display title
st.title("Local Recommender Demo")

# Set initial post
if "current_post" not in st.session_state:
    st.session_state.current_post = generate_post()

post = st.session_state.current_post

# Initialize widget session states
if "liked_input" not in st.session_state:
    st.session_state.liked_input = False
if "commented_input" not in st.session_state:
    st.session_state.commented_input = False
if "interested_input" not in st.session_state:
    st.session_state.interested_input = False
if "not_interested_input" not in st.session_state:
    st.session_state.not_interested_input = False
if "watchtime_slider" not in st.session_state:
    st.session_state.watchtime_slider = st.session_state.current_post['duration'] / 2

st.subheader(f"Topic: {post['topic']}")
st.text(f"Hashtags: {' '.join(post['hashtags'])}")
st.text(f"Video Duration: {post['duration']} seconds")

# Interactive inputs
col1, col2, col3, col4 = st.columns(4)
liked = col1.checkbox("Like", value=st.session_state.liked_input, key="liked_input")
commented = col2.checkbox("Comment", value=st.session_state.commented_input, key="commented_input")
interested = col3.checkbox("Interested", value=st.session_state.interested_input, key="interested_input")
not_interested = col4.checkbox("Not Interested", value=st.session_state.not_interested_input, key="not_interested_input")

time_watched = st.slider(
    "How long did you watch this post?",
    0.0, post['duration'] * 2,
    value=st.session_state.watchtime_slider,
    key="watchtime_slider"
)

# Next button
next_clicked = st.button("Next")

# Save interaction and load new post on next
if next_clicked:
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

    # Clear widget states on rerun
    for key in ["liked_input", "commented_input", "interested_input", "not_interested_input", "watchtime_slider"]:
        if key in st.session_state:
            del st.session_state[key]

    try:
        st.experimental_rerun()
    except:
        st.stop()

# Display interaction history and predictions
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
