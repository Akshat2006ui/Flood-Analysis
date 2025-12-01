import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Flood Social Media Sentiment Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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


def get_sentiment(text):
    """
    Analyze sentiment of text using TextBlob with enhanced accuracy
    Returns: polarity (-1 to 1) and sentiment label
    """
    if pd.isna(text) or text == '' or text is None:
        return 0, 'neutral', 0
    
    try:
        text = str(text).strip()
        if len(text) < 3:
            return 0, 'neutral', 0

        # Cleaning
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()

        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity

        # Categorization
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


def load_and_clean_data_auto():
    """Auto-load dataset flood_social_data.csv"""
    try:
        df = pd.read_csv("flood_social_data.csv", encoding="utf-8")

        blank_columns = df.columns[df.isnull().all()].tolist()
        if blank_columns:
            df = df.drop(columns=blank_columns)

        df = df.dropna(subset=['content'])

        return df, True
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, False


def create_sentiment_summary(df_clean):
    summary = {
        'total_posts': len(df_clean),
        'positive_count': (df_clean['sentiment_label'].isin(['positive', 'slightly_positive'])).sum(),
        'negative_count': (df_clean['sentiment_label'].isin(['negative', 'slightly_negative'])).sum(),
        'neutral_count': (df_clean['sentiment_label'] == 'neutral').sum(),
        'avg_polarity': df_clean['sentiment_polarity'].mean(),
        'avg_subjectivity': df_clean['sentiment_subjectivity'].mean(),
        'sentiment_distribution': df_clean['sentiment_label'].value_counts()
    }
    return summary


def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Flood Social Media Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("Navigation & Controls")
    st.sidebar.markdown("---")
    st.sidebar.info("Dataset auto-loaded: **flood_social_data.csv**")

    # Auto load dataset
    df_clean, success = load_and_clean_data_auto()

    if not success:
        st.stop()

    # Sidebar options
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    show_sentiment_stats = st.sidebar.checkbox("Show Sentiment Statistics", value=True)
    advanced_analysis = st.sidebar.checkbox("Advanced Analysis", value=True)

    # Perform sentiment analysis
    with st.spinner("Performing sentiment analysis..."):
        sentiment_results = df_clean['content'].apply(get_sentiment)
        df_clean['sentiment_polarity'] = sentiment_results.apply(lambda x: x[0])
        df_clean['sentiment_label'] = sentiment_results.apply(lambda x: x[1])
        df_clean['sentiment_subjectivity'] = sentiment_results.apply(lambda x: x[2])

    # Summary
    summary = create_sentiment_summary(df_clean)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📈 Sentiment Analysis", "🌍 Geographical",
        "📱 Sources", "📅 Temporal", "🔍 Data Explorer"
    ])

    # --- TAB 1 ---
    with tab1:
        st.header("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Posts", summary['total_posts'])
        col2.metric("Positive", summary['positive_count'])
        col3.metric("Negative", summary['negative_count'])
        col4.metric("Avg Polarity", f"{summary['avg_polarity']:.3f}")

        # Missing values
        missing = df_clean.isnull().sum()
        missing = missing[missing > 0]

        col1, col2 = st.columns(2)
        with col1:
            if len(missing) > 0:
                fig = px.bar(missing, x=missing.index, y=missing.values, title="Missing Values")
                st.plotly_chart(fig)
            else:
                st.success("No missing values")

        with col2:
            df_clean['content_length'] = df_clean['content'].str.len()
            fig = px.histogram(df_clean, x='content_length', nbins=50, title="Content Length Distribution")
            st.plotly_chart(fig)

        if show_raw_data:
            st.subheader("Raw Data Preview")
            st.dataframe(df_clean.head(100))

    # --- TAB 2 ---
    with tab2:
        st.header("📈 Sentiment Analysis")

        if show_sentiment_stats:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strong Positive", len(df_clean[df_clean['sentiment_label'] == 'positive']))
            col2.metric("Strong Negative", len(df_clean[df_clean['sentiment_label'] == 'negative']))
            col3.metric("Avg Subjectivity", f"{summary['avg_subjectivity']:.3f}")
            col4.metric("Polarity Std Dev", f"{df_clean['sentiment_polarity'].std():.3f}")

        # Sentiment distribution
        sentiment_map = {
            'positive': 'Positive',
            'slightly_positive': 'Slightly Positive',
            'negative': 'Negative',
            'slightly_negative': 'Slightly Negative',
            'neutral': 'Neutral'
        }

        df_clean['sentiment_category'] = df_clean['sentiment_label'].map(sentiment_map)
        counts = df_clean['sentiment_category'].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=counts.values, names=counts.index,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig)

        with col2:
            fig = px.scatter(
                df_clean, x='sentiment_polarity', y='sentiment_subjectivity',
                color='sentiment_category', hover_data=['content'],
                title='Polarity vs Subjectivity'
            )
            st.plotly_chart(fig)

        st.subheader("Detailed Sentiment Statistics")
        stats = df_clean.groupby('sentiment_category')['sentiment_polarity'].agg(['count', 'mean', 'std'])
        st.dataframe(stats)

    # --- TAB 3 ---
    with tab3:
        st.header("🌍 Geographical Analysis")
        if 'userLocation' in df_clean.columns and advanced_analysis:
            loc = df_clean.groupby('userLocation').agg({
                'sentiment_polarity': ['mean', 'count'],
            }).dropna()
            loc.columns = ['avg_polarity', 'post_count']

            st.dataframe(loc.sort_values('post_count', ascending=False))

            fig = px.bar(loc.nlargest(10, 'post_count'), y='post_count', title="Top Locations")
            st.plotly_chart(fig)

        else:
            st.info("No location data")

    # --- TAB 4 ---
    with tab4:
        st.header("📱 Source Platform Analysis")
        if 'source' in df_clean.columns:
            src = df_clean.groupby('source').agg({
                'sentiment_polarity': ['mean', 'count']
            })
            src.columns = ['avg_polarity', 'post_count']

            fig = px.bar(src, y='post_count', title="Posts by Source")
            st.plotly_chart(fig)

            st.dataframe(src)
        else:
            st.info("No source column")

    # --- TAB 5 ---
    with tab5:
        st.header("📅 Temporal Analysis")
        date_cols = [c for c in df_clean.columns if "date" in c.lower() or "time" in c.lower()]

        if date_cols:
            date_col = st.selectbox("Select date column", date_cols)
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_temp = df_clean.dropna(subset=[date_col]).set_index(date_col)

            resampled = df_temp.resample("D").agg({
                'sentiment_polarity': 'mean',
                'content': 'count'
            })

            fig = px.line(resampled, y='sentiment_polarity', title="Daily Sentiment Trend")
            st.plotly_chart(fig)

            fig = px.line(resampled, y='content', title="Daily Post Count")
            st.plotly_chart(fig)

        else:
            st.info("No date/time column")

    # --- TAB 6 ---
    with tab6:
        st.header("🔍 Data Explorer")

        sentiment_options = ['Positive', 'Slightly Positive', 'Neutral', 'Slightly Negative', 'Negative']
        choice = st.multiselect("Sentiment Filter", sentiment_options, default=sentiment_options)

        mapping = {
            'Positive': 'positive',
            'Slightly Positive': 'slightly_positive',
            'Neutral': 'neutral',
            'Slightly Negative': 'slightly_negative',
            'Negative': 'negative'
        }

        selected = [mapping[x] for x in choice]

        filtered = df_clean[df_clean['sentiment_label'].isin(selected)]

        st.dataframe(filtered.head(200))

        st.subheader("Sample Content")
        for i, row in filtered.head(10).iterrows():
            with st.expander(f"Polarity: {row['sentiment_polarity']:.3f}"):
                st.write(row['content'])

    # Export
    st.sidebar.markdown("---")
    csv = df_clean.to_csv(index=False)
    st.sidebar.download_button(
        "📥 Download Analyzed CSV",
        csv,
        "flood_sentiment_analysis.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()
