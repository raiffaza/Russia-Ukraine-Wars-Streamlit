import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests

# Function to download the dataset from Google Drive
@st.cache_data
def download_data():
    # Google Drive file ID (extracted from your link)
    file_id = '1GvHWbdGp8MV2XzMZuMY1M8PuB6xDX4gc'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # Send GET request to download the file
    response = requests.get(url, stream=True)
    with open('dataset_with_sentiment.csv', 'wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)

    # Load the dataset after downloading
    df = pd.read_csv('dataset_with_sentiment.csv')
    # Convert 'post_created_time' to datetime
    df['post_created_time'] = pd.to_datetime(df['post_created_time'])
    # Standardize 'side' column to title case
    df['side'] = df['side'].str.title()
    # Correct misspellings in the 'side' column
    df['side'] = df['side'].replace({"Rusia": "Russia", "Ukraina": "Ukraine"})
    return df

# Load the data
df = download_data()

# Streamlit Info
st.info("The source code for this Streamlit app is available on [GitHub](https://github.com/raiffaza/RUSSIA-UKRAINE-REDDIT-USERS-SENTIMENT).")

# Sidebar filters
st.sidebar.header("Filters")

# Component 1: Sidebar Dropdown for Side
selected_side = st.sidebar.selectbox("Select Side", ["All", "Russia", "Ukraine", "USA"])

# Component 2: Multiselect for Sentiment
selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment", 
    ["Positive", "Neutral", "Negative"], 
    default=["Positive", "Neutral", "Negative"]
)

# Component 3: Date Range Slider
min_date = df['post_created_time'].min().date()
max_date = df['post_created_time'].max().date()
date_range = st.sidebar.slider(
    "Select Date Range", 
    min_value=min_date, 
    max_value=max_date, 
    value=(min_date, max_date)
)

# Component 4: Text Input for Keyword Search
keyword = st.sidebar.text_input("Search Keyword in Comments", "")

# Filter data based on user inputs
filtered_df = df[df['sentiment'].isin(selected_sentiment)]
if selected_side != "All":
    filtered_df = filtered_df[filtered_df['side'] == selected_side]

filtered_df = filtered_df[
    (filtered_df['post_created_time'].dt.date >= date_range[0]) & 
    (filtered_df['post_created_time'].dt.date <= date_range[1])
]

if keyword:
    filtered_df = filtered_df[filtered_df['clean_text'].str.contains(keyword, case=False)]

# Title and description
st.title("Public Sentiment Analysis on Russia-Ukraine Conflict")
st.write("""
Analyze sentiment trends and topics discussed on Reddit regarding the Russia-Ukraine conflict. 
This analysis examines public sentiment following the U.S. presidential election in November 2024, focusing on comments from Reddit users.
""")

# Sentiment Distribution (Pie Chart)
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_df['sentiment'].value_counts()
if not sentiment_counts.empty:
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
else:
    st.write("No data available for the selected filters.")

# Sentiment Trends Over Time (Line Chart)
st.subheader("Sentiment Trends Over Time")
filtered_df['date'] = filtered_df['post_created_time'].dt.date
trends = filtered_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
if not trends.empty:
    st.line_chart(trends)
else:
    st.write("No data available for the selected filters.")

# Word Cloud
st.subheader("Word Cloud")
text = " ".join(filtered_df['clean_text'].dropna())
if text:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("No text data available for the selected filters.")

# Raw Data Preview
st.subheader("Raw Data Preview")
st.dataframe(filtered_df.head(10))

# Download Button for Filtered Data
if not filtered_df.empty:
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(filtered_df)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )
