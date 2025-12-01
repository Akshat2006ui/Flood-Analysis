import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Flood Social Media Sentiment Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 10px 0;
}
.positive { color: #2ecc71; }
.negative { color: #e74c3c; }
.neutral { color: #3498db; }
</style>
""", unsafe_allow_html=True)

# ------------------- SENTIMENT FUNCTION -------------------
def get_sentiment(text):
    if pd.isna(text) or text == '' or text is None:
        return 0, 'neutral', 0
    
    try:
        text = str(text).strip()
        if len(text) < 3:
            return 0, 'neutral', 0

        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()

        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        if subjectivity < 0.1:
            sentiment = 'neutral'
        elif polarity > 0.2:
            sentiment = 'positive'
        elif polarity > 0.05:
            sentiment = 'slightly_positive'
        elif polarity < -0.2:
            sentiment = 'negative'
        elif polarity < -0.05:
            sentiment = 'slightly_negative'
        else:
            sentiment = 'neutral'

        return polarity, sentiment, subjectivity

    except:
        return 0, 'neutral', 0

# ------------------- CACHED DATA LOAD -------------------
@st.cache_data
def load_clean_data():
    df = pd.read_csv("flood_social_data.csv", encoding="utf-8")
    blank_columns = df.columns[df.isnull().all()].tolist()
    if blank_columns:
        df = df.drop(columns=blank_columns)
    df = df.dropna(subset=["content"])
    return df

# ------------------- SUMMARY -------------------
def create_summary(df):
    return {
        'total_posts': len(df),
        'positive_count': (df['sentiment_label'].isin(['positive', 'slightly_positive'])).sum(),
        'negative_count': (df['sentiment_label'].isin(['negative', 'slightly_negative'])).sum(),
        'neutral_count': (df['sentiment_label'] == 'neutral').sum(),
        'avg_polarity': df['sentiment_polarity'].mean(),
        'avg_subjectivity': df['sentiment_subjectivity'].mean(),
        'dist': df['sentiment_label'].value_counts()
    }

# ------------------- MAIN APP -------------------
def main():

    st.markdown('<h1 class="main-header">🌊 Flood Social Media Sentiment Analysis</h1>', unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    st.sidebar.info("Dataset auto loaded: flood_social_data.csv")

    df = load_clean_data()

    show_raw = st.sidebar.checkbox("Show Raw Data", False)

    with st.spinner("Performing sentiment analysis..."):
        results = df["content"].apply(get_sentiment)
        df["sentiment_polarity"] = results.apply(lambda x: x[0])
        df["sentiment_label"] = results.apply(lambda x: x[1])
        df["sentiment_subjectivity"] = results.apply(lambda x: x[2])

    summary = create_summary(df)

    # Lazy imports for heavy libs
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import seaborn as sns

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Sentiment", "🌍 Geography",
        "📱 Sources", "📅 Temporal", "🔍 Explorer"
    ])

    # TAB 1 ----------------------
    with tab1:
        st.header("Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Posts", summary['total_posts'])
        c2.metric("Positive", summary['positive_count'])
        c3.metric("Negative", summary['negative_count'])
        c4.metric("Avg Polarity", f"{summary['avg_polarity']:.3f}")

        df["content_length"] = df["content"].str.len()
        fig = px.histogram(df, x="content_length", nbins=40, title="Content Length Distribution")
        st.plotly_chart(fig)

        if show_raw:
            st.dataframe(df.head(200))

    # TAB 2 ----------------------
    with tab2:
        st.header("📈 Sentiment Analysis")

        mapping = {
            'positive': 'Positive',
            'slightly_positive': 'Slightly Positive',
            'negative': 'Negative',
            'slightly_negative': 'Slightly Negative',
            'neutral': 'Neutral'
        }
        df["sent_cat"] = df["sentiment_label"].map(mapping)

        fig = px.pie(df["sent_cat"].value_counts(), title="Sentiment Distribution")
        st.plotly_chart(fig)

        fig = px.scatter(
            df, x="sentiment_polarity", y="sentiment_subjectivity",
            color="sent_cat", hover_data=["content"],
            title="Polarity vs Subjectivity"
        )
        st.plotly_chart(fig)

    # TAB 3 ----------------------
    with tab3:
        st.header("🌍 Geographical Analysis")
        if "userLocation" in df.columns:
            loc = df.groupby("userLocation").agg({"sentiment_polarity":["mean","count"]})
            st.dataframe(loc)
        else:
            st.info("No location column found")

    # TAB 4 ----------------------
    with tab4:
        st.header("📱 Source Analysis")
        if "source" in df.columns:
            src = df.groupby("source")["content"].count()
            fig = px.bar(src, title="Posts by Source")
            st.plotly_chart(fig)
        else:
            st.info("No source column")

    # TAB 5 ----------------------
    with tab5:
        st.header("📅 Temporal Analysis")

        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols:
            col = st.selectbox("Choose date column", date_cols)
            df[col] = pd.to_datetime(df[col], errors="ignore")
            df2 = df.dropna(subset=[col]).set_index(col)
            res = df2.resample("D").count()["content"]
            fig = px.line(res, title="Daily Post Count")
            st.plotly_chart(fig)
        else:
            st.info("No date/time column")

    # TAB 6 ----------------------
    with tab6:
        st.header("🔍 Data Explorer")

        options = ["Positive", "Slightly Positive", "Neutral", "Slightly Negative", "Negative"]
        choice = st.multiselect("Filter by sentiment", options, options)

        reverse_map = {
            "Positive": "positive",
            "Slightly Positive": "slightly_positive",
            "Neutral": "neutral",
            "Slightly Negative": "slightly_negative",
            "Negative": "negative"
        }
        selected = [reverse_map[x] for x in choice]

        filtered = df[df["sentiment_label"].isin(selected)]
        st.dataframe(filtered.head(200))

    # DOWNLOAD ----------------------
    st.sidebar.download_button(
        "📥 Download Analyzed CSV",
        df.to_csv(index=False),
        "flood_sentiment_analysis.csv",
        "text/csv"
    )

if __name__ == "__main__":
    main()
