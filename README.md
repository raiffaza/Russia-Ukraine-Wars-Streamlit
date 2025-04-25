# Russia-Ukraine Reddit Sentiment Analysis

## Overview
This Streamlit app analyzes public sentiment on Reddit regarding the Russia-Ukraine conflict, focusing on trends, key events, and public opinion shifts following the U.S. presidential election in November 2024. The analysis uses data from Kaggle and employs VADER and Hugging Face models for sentiment analysis.

The dataset has been randomly sampled to ensure its size is under **24 MB**, making it compatible with GitHub's file size restrictions. This smaller dataset retains the essential features and structure of the original dataset while being lightweight and manageable.

---

## Features
- **Filter Data**: Filter by side (Russia, Ukraine, USA), sentiment (Positive, Neutral, Negative), date range, and keyword search.
- **Visualizations**:
  - Pie charts showing sentiment distribution.
  - Line charts tracking sentiment trends over time.
  - Word clouds highlighting frequently mentioned terms.
- **Download Filtered Data**: Export filtered data for further analysis.

---

## Dataset
- **Source**: Kaggle dataset titled "Public Opinion on the Russia-Ukraine War (Updated Daily)".
- **Features**:
  - Comments, sentiment scores, side (Russia, Ukraine, USA), and timestamps.
  - Preprocessing includes sentiment analysis using VADER and Hugging Face models, data cleaning, and feature engineering.
- **Sampling**: The dataset has been randomly sampled to under **24 MB** to comply with GitHub's file size limits. This ensures the dataset remains accessible while preserving its representativeness.

---

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/raiffaza/Russia-Ukraine-Wars-Streamlit.git
