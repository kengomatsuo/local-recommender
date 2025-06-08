# model.py
from keybert import KeyBERT
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class KeyBERTVectorizer:
    def __init__(self, model='all-MiniLM-L6-v2', n_keywords=5):
        self.kw_model = KeyBERT(model=model)
        self.n_keywords = n_keywords
        self.cache = {}

    def transform(self, texts):
        keywords_list = []
        for text in texts:
            if text in self.cache:
                keywords = self.cache[text]
            else:
                keywords = self.kw_model.extract_keywords(text, top_n=self.n_keywords)
                self.cache[text] = keywords
            keywords_list.append(" ".join([kw[0] for kw in keywords]))
        return keywords_list

    def fit(self, texts, y=None):
        return self

class LocalRecommenderClassifier:
    def __init__(self):
        self.pipeline = None
        self.trained = False
        self.last_trained_on = 0

    def fit(self, df_user: pd.DataFrame):
        df = df_user[df_user["engaged"] != 1].copy()
        df["liked"] = df["liked"].astype(int)
        df["commented"] = df["commented"].astype(int)
        df["hashtags_str"] = df["hashtags"].apply(lambda x: " ".join(x))

        X = df[["topic", "liked", "commented", "duration", "time_watched", "hashtags_str", "caption"]]
        y = df["engaged"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("topic", HashingVectorizer(n_features=30), "topic"),
                ("hashtags", HashingVectorizer(n_features=30), "hashtags_str"),
                ("caption_tfidf", TfidfVectorizer(stop_words='english', max_features=30), "caption"),
                ("caption_kw", Pipeline([
                    ('kw', KeyBERTVectorizer(n_keywords=3)),
                    ('vec', HashingVectorizer(n_features=30))
                ]), "caption"),
                ("num", StandardScaler(), ["liked", "commented", "duration", "time_watched"]),
            ]
        )

        self.pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42))
        ])

        self.pipeline.fit(X, y)
        self.trained = True
        self.last_trained_on = len(df_user)

    def recommend(self, df_user: pd.DataFrame, normalize=True):
        if not self.trained:
            return {}, {}

        df = df_user.copy()
        df["liked"] = df["liked"].astype(int)
        df["commented"] = df["commented"].astype(int)
        df["hashtags_str"] = df["hashtags"].apply(lambda x: " ".join(x))

        X_all = df[["topic", "liked", "commented", "duration", "time_watched", "hashtags_str", "caption"]]
        probas_all = self.pipeline.predict_proba(X_all)

        df["pred_proba"] = probas_all[:, 1] if probas_all.shape[1] > 1 else 0

        topic_scores = df.groupby("topic")["pred_proba"].mean().round(3).to_dict()

        all_tags = set(tag for tags in df["hashtags"] for tag in tags)
        hashtag_scores = {}
        for tag in all_tags:
            scores = df[df["hashtags"].apply(lambda tags: tag in tags)]["pred_proba"]
            if not scores.empty:
                hashtag_scores[tag] = round(scores.mean(), 3)

        if normalize:
            t_total = sum(topic_scores.values())
            h_total = sum(hashtag_scores.values())
            topic_scores = (
                {k: round(v / t_total, 3) for k, v in topic_scores.items()} if t_total > 0 else {}
            )
            hashtag_scores = (
                {k: round(v / h_total, 3) for k, v in hashtag_scores.items()} if h_total > 0 else {}
            )

        return topic_scores, hashtag_scores
