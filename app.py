import streamlit as st
import pandas as pd
import time
import tracemalloc
from utils import generate_post
from model import LocalRecommenderClassifier
from sklearn.metrics import classification_report

st.title("Local Recommender with Hashtags and Performance Stats")

# Initialize session state variables
if "interactions" not in st.session_state:
    st.session_state.interactions = []
if "current_post" not in st.session_state:
    st.session_state.current_post = generate_post()
if "post_counter" not in st.session_state:
    st.session_state.post_counter = 0
if "model" not in st.session_state:
    st.session_state.model = LocalRecommenderClassifier()
if "fit_cooldown" not in st.session_state:
    st.session_state.fit_cooldown = 0
if "model_processing" not in st.session_state:
    st.session_state.model_processing = False

if st.session_state.fit_cooldown > 0:
    st.info(
        f"Model fitting is on cooldown for {st.session_state.fit_cooldown} more post(s).")

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
        (time_watched / post["duration"]
         ) < 0.25 and not liked and not commented
    ):
        engaged = 0
    else:
        engaged = int(
            (time_watched / post["duration"]) > 0.8 or liked or commented) + 1

    st.session_state.interactions.append({
        "topic": post["topic"],
        "hashtags": post["hashtags"],
        "liked": liked,
        "commented": commented,
        "time_watched": time_watched,
        "duration": post["duration"],
        "caption": post["caption"],
        "engaged": engaged,
    })

    st.session_state.current_post = generate_post()
    st.session_state.post_counter += 1
    st.session_state.liked_input = False
    st.session_state.commented_input = False
    st.session_state.interest_input = "Neutral"

    if st.session_state.fit_cooldown > 0:
        st.session_state.fit_cooldown -= 1


post = st.session_state.current_post
post_key = str(st.session_state.post_counter)

st.subheader(f"Topic: {post['topic']}")
st.text(f"Hashtags: {' '.join(post['hashtags'])}")
st.text(f"Caption: {post['caption']}")
st.text(f"Video Duration: {post['duration']} seconds")

liked = st.checkbox("Liked", key=f"liked_input",
                    disabled=st.session_state.model_processing)
commented = st.checkbox("Commented", key=f"commented_input",
                        disabled=st.session_state.model_processing)
interest_flag = st.radio(
    "Interest Feedback",
    options=["Neutral", "Interested", "Not Interested"],
    index=0,
    key=f"interest_input",
    disabled=st.session_state.model_processing
)
time_watched = st.slider(
    "Time Watched", 0.0, post["duration"] * 2, post["duration"] / 2,
    key=f"watch_input_{st.session_state.post_counter}",
    disabled=st.session_state.model_processing
)

st.button("Next", on_click=handle_next_click,
          disabled=st.session_state.model_processing)

if st.session_state.interactions:
    df = pd.DataFrame(st.session_state.interactions)
    st.subheader("User Interactions")
    st.dataframe(df)

    enough_data = len(df[df["engaged"] != 1]) >= 10
    model = st.session_state.model

    if enough_data:
        
        if st.session_state.fit_cooldown == 0 and (not model.trained or len(df) - model.last_trained_on >= 5):
            if (not st.session_state.model_processing):
                st.session_state.model_processing = True
                st.rerun()
            else:
                st.info("Model fitting. Please wait. This will take a few seconds.")
                
                model_start = time.perf_counter()
                tracemalloc.start()
                # print(df)
                model.fit(df)
                current, peak = tracemalloc.get_traced_memory()
                model_time = time.perf_counter() - model_start
                
                topic_weights, hashtag_weights = model.recommend(df)
                tracemalloc.stop()
                
                confusion_matrix = st.session_state.model.confusion_matrix
                classification_report = st.session_state.model.classification_report


                
                # print(model.recommend(df, normalize=True))
                # print(topic_weights, hashtag_weights)
                # print(current, peak)
                print(f"Confusion Matrix:\n{confusion_matrix}")
                print(f"Classification Report:\n{classification_report}")
                
                st.session_state.model_time = model_time
                st.session_state.model_memory = (current, peak)
                st.session_state.topic_weights = topic_weights
                st.session_state.hashtag_weights = hashtag_weights
                st.session_state.confusion_matrix = confusion_matrix
                st.session_state.classification_report = classification_report
                
                st.session_state.model_processing = False
                st.session_state.fit_cooldown = 10
                st.rerun()
        
        st.subheader("Inferred Topic Preferences")
        topic_weights = st.session_state.topic_weights
        if topic_weights:
            st.bar_chart(pd.DataFrame(topic_weights.items(), columns=[
                         "Topic", "Weight"]).set_index("Topic"))

        st.subheader("Inferred Hashtag Preferences")
        hashtag_weights = st.session_state.hashtag_weights
        if hashtag_weights:
            st.bar_chart(pd.DataFrame(hashtag_weights.items(), columns=[
                         "Hashtag", "Weight"]).set_index("Hashtag"))

        st.subheader("Performance Statistics")
        model_time = st.session_state.model_time or 0.0
        st.markdown(f"**Model Inference Time:** {model_time:.4f} seconds")
        current, peak = st.session_state.model_memory or (0, 0)
        st.markdown(
            f"**Model Memory Usage:** {current / 1024:.2f} KB (current), {peak / 1024:.2f} KB (peak)")

        st.subheader("Confusion Matrix")
        confusion_matrix = st.session_state.confusion_matrix
        st.dataframe(confusion_matrix)
        st.subheader("Classification Report")
        classification_report = st.session_state.classification_report
        if classification_report:
            report_df = pd.DataFrame(classification_report).transpose()
            st.dataframe(report_df)