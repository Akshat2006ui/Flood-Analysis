## 🌊 Flood Social Media Sentiment Analysis Dashboard
A Streamlit + NLP Project for Real-Time Public Sentiment Monitoring During Floods
📌 Overview

This project provides an interactive, real-time sentiment analysis dashboard for analyzing public emotions during flood situations using social media data.
It combines NLP, data visualization, and Streamlit UI to generate insights into how people react, respond, and communicate during flood emergencies.

The application automatically processes a dataset named flood_social_data.csv and visualizes sentiment, trends, locations, and more through a multi-tab dashboard.

## 🧠 Tech Stack

NLP: TextBlob

Frontend/UI: Streamlit

Visualization: Plotly, Matplotlib, Seaborn

Data Processing: Pandas, NumPy

Preprocessing: Regex

## 🎯 Key Features
🔹 1. Automated Data Loading & Text Cleaning

✔ Loads flood_social_data.csv on startup
✔ Removes unwanted/blank columns
✔ Cleans:

URLs
Hashtags
Mentions
Emojis

Punctuation
✔ Handles missing values
✔ Creates derived columns (length, polarity, subjectivity, classification)

🔹 2. Advanced Sentiment Analysis

Uses TextBlob polarity + subjectivity to classify tweets into five categories:

## Polarity Range	Label

polarity > 0.20	Positive
0.05 < polarity ≤ 0.20	Slightly Positive
−0.05 ≤ polarity ≤ 0.05	Neutral
−0.20 ≤ polarity < −0.05	Slightly Negative
polarity < −0.20	Negative

## ⚠ Additional rule:
If subjectivity < 0.10, the post is automatically marked as Neutral.

🔹 3. Fully Interactive Streamlit Dashboard

The UI contains multiple analysis tabs:

## Tab	Description
📊 Overview	Dataset info, missing values, content length distribution
📈 Sentiment Analysis	Pie charts, scatter plots, sentiment stats
🌍 Geographical Insights	Location-wise polarity maps & bar charts
📱 Source-Based Trends	Platform-wise activity (Twitter, FB, etc.)
📅 Temporal Analysis	Daily trends, sentiment over time
🔍 Data Explorer	Filter posts by sentiment & inspect raw text
📁 Folder Structure
📦 flood-sentiment-analysis
│
├── app.py                      # Main Streamlit application
├── flood_social_data.csv       # Input dataset
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies

## 🛠 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/flood-sentiment-analysis.git
cd flood-sentiment-analysis

2️⃣ Install Dependencies

Create requirements.txt with:

streamlit
pandas
numpy
textblob
regex
plotly
matplotlib
seaborn


## Install packages:

pip install -r requirements.txt
python -m textblob.download_corpora

3️⃣ Add Your Dataset

Place your CSV file as:

flood_social_data.csv

4️⃣ Run the Application
streamlit run app.py

## 📊 Dataset Requirements
Column	Description
content	Text of social media post
date or time (optional)	For temporal analysis
userLocation (optional)	For geographical analysis
source (optional)	Platform name (Twitter, Facebook, etc.)


🧪 Sentiment Classification Logic
polarity > 0.20                → Positive  
0.05 < polarity ≤ 0.20         → Slightly Positive  
−0.05 ≤ polarity ≤ 0.05        → Neutral  
−0.20 ≤ polarity < −0.05       → Slightly Negative  
polarity < −0.20               → Negative  

subjectivity < 0.10            → Neutral (override rule)

## 📥 Export Feature

The dashboard allows exporting a fully processed dataset:

flood_sentiment_analysis.csv

🚀 Deployment Options
✔ Streamlit Cloud (Recommended)

Push repo → Deploy → Share public link
Fully serverless & auto-scaling.

✔ HuggingFace Spaces

Create README.md with:

title: Flood Sentiment Analysis
emoji: 🌊
sdk: streamlit
sdk_version: 1.26.0

✔ Local Deployment

Runs on Windows, macOS, and Linux.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork the repo and submit a pull request.

## 📜 License

This project is licensed under the MIT License.

## 🔗 Live Project
🚀 Live Dashboard:
👉 https://flood-analysis-wzhbjuvcdhjrlwxerhfjam.streamlit.app/

## AKSHAT KAPOOR
## (AI ENGINEER)
