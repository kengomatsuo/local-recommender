# Local Recommender App

This is a Streamlit-based local recommender system that uses hashtags and performance stats to recommend posts.

## Requirements
- Python 3.11 or newer
- All dependencies in `requirements.txt`

## Setup
1. **Install dependencies**
   Open a terminal in the project directory and run:
   
   ```powershell
   pip install -r requirements.txt
   ```

2. **Ensure `posts.csv` is present**
   The app uses `posts.csv` as its data source. Make sure this file exists in the project root.

## Running the App
Start the Streamlit app with:

```powershell
streamlit run app.py
```

This will open the app in your default web browser.

## Files
- `app.py`: Main Streamlit app
- `model.py`: Model and vectorizer definitions
- `utils.py`: Utility functions for data and post generation
- `posts.csv`: Data file with posts
- `requirements.txt`: Python dependencies

## Notes
- The app uses session state to track user interactions.
- Model fitting and recommendations are handled in `model.py`.
