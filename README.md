🌊 Flood Social Media Sentiment Analysis Dashboard

A Streamlit + NLP Project for Analyzing Public Sentiment During Flood Situations

📌 Project Overview

This project provides an interactive, real-time sentiment analysis dashboard for flood-related social media data.
It uses:

TextBlob for sentiment polarity & subjectivity

Streamlit for UI and live visualization

Plotly for advanced interactive charts

Pandas/NumPy for data processing

Regex cleaning for preprocessing text

The application automatically loads a dataset named flood_social_data.csv and generates a fully interactive dashboard with multiple analysis tabs.

🎯 Key Features

🔹 1. Automatic Data Loading & Cleaning

Loads flood_social_data.csv automatically

Removes blank columns

Cleans URLs, hashtags, mentions, and punctuation

Handles missing values

🔹 2. Advanced Sentiment Analysis

Custom TextBlob-based classifier:

Positive

Slightly Positive

Neutral

Slightly Negative

Negative

Outputs include:

Polarity (−1 to 1)

Subjectivity

Cleaned sentiment label

🔹 3. Interactive Dashboard Tabs
Tab	Description
📊 Overview	Dataset info, missing values, content length distribution
📈 Sentiment Analysis	Pie chart, scatter plot, sentiment stats
🌍 Geographical	Location-wise polarity & counts
📱 Source	Source platform analysis
📅 Temporal	Time-based trends (daily sentiment, frequency)
🔍 Data Explorer	Filter posts by sentiment and explore content
📁 Folder Structure
📦 flood-sentiment-analysis
│
├── app.py                      # Main Streamlit application
├── flood_social_data.csv       # Input social media dataset
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies

🛠 Installation & Setup
1. Clone the Repository
git clone https://github.com/yourusername/flood-sentiment-analysis.git
cd flood-sentiment-analysis

2. Install Required Packages

Create requirements.txt with:

streamlit
pandas
numpy
textblob
regex
plotly
matplotlib
seaborn


Install dependencies:

pip install -r requirements.txt
python -m textblob.download_corpora

3. Add Your Dataset

Place flood_social_data.csv in the project folder.

4. Run the Streamlit App
streamlit run app.py

📊 Dataset Requirements

The CSV must have at least:

Column	Description
content	Social media text
date or time (optional)	For temporal trends
userLocation (optional)	For geographical analysis
source (optional)	Platform (Twitter, FB, etc.)
🧪 Sentiment Classification Logic
Criteria	Label
polarity > 0.20	Positive
0.05 < polarity ≤ 0.20	Slightly Positive
−0.05 ≤ polarity ≤ 0.05	Neutral
−0.20 ≤ polarity < −0.05	Slightly Negative
polarity < −0.20	Negative

Subjectivity < 0.1 is always marked as neutral. 

📥 Export Feature

You can download a fully processed CSV:

flood_sentiment_analysis.csv

🚀 Deployment Options

You can deploy this project on:

✔ Streamlit Cloud

Just push repo → click “Deploy”
Supports autoscaling and public sharing.

✔ HuggingFace Spaces

Add:

title: Flood Sentiment Analysis
emoji: 🌊
sdk: streamlit
sdk_version: 1.26.0

✔ Local Deployment

Works on any OS with Python installed.

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this project and submit a pull request.

📜 License

This project is licensed under the MIT License.

🔗 Live Project

🚀 Live Dashboard:

Link:https://flood-analysis-wzhbjuvcdhjrlwxerhfjam.streamlit.app/
