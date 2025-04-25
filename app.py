import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_with_sentiment.csv")
    # Convert 'post_created_time' to datetime
    df['post_created_time'] = pd.to_datetime(df['post_created_time'])
    # Standardize 'side' column to title case
    df['side'] = df['side'].str.title()
    # Correct misspellings in the 'side' column
    df['side'] = df['side'].replace({"Rusia": "Russia", "Ukraina": "Ukraine"})
    return df

df = load_data()
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
    # Debug: Print unique sides in the dataset
    print(f"Unique sides in dataset: {df['side'].unique()}")
    print(f"Selected side: {selected_side}")
    # Filter by side
    filtered_df = filtered_df[filtered_df['side'] == selected_side]
    print(f"Filtered by side: {filtered_df.shape}")

filtered_df = filtered_df[
    (filtered_df['post_created_time'].dt.date >= date_range[0]) & 
    (filtered_df['post_created_time'].dt.date <= date_range[1])
]
print(f"Filtered by date range: {filtered_df.shape}")

if keyword:
    filtered_df = filtered_df[filtered_df['clean_text'].str.contains(keyword, case=False)]
    print(f"Filtered by keyword: {filtered_df.shape}")

# Title and description
st.title("Public Sentiment Analysis on Russia-Ukraine Conflict")
st.write("""
Analyze sentiment trends and topics discussed on Reddit regarding the Russia-Ukraine conflict. 
This analysis examines public sentiment following the U.S. presidential election in November 2024, focusing on comments from Reddit users.
""")

# Explanation Section
with st.expander("About the Research"):
    st.markdown("""
    #### Background
    - Since November 5, 2024, former President Donald Trump has consistently expressed concerns over U.S. funding for military aid to Ukraine, emphasizing his priority of addressing domestic issues.
    - This debate escalated on February 28, 2025, when President Trump reacted strongly to President Zelensky's decision to reject a U.S.- or Russia-brokered peace or ceasefire agreement.
    - For this analysis, we utilized comments from Reddit users to assess sentiment toward Russia and Ukraine from November 5, 2024, to the latest available date in a dataset sourced from Kaggle, titled "Public Opinion on the Russia-Ukraine War (Updated Daily)."

    #### Objectives
    - Examine sentiment trends over time.
    - Identify key events that triggered changes in sentiment.
    - Provide insights into public opinion regarding the geopolitical landscape.

    #### Dataset
    - Source: Kaggle dataset titled "Public Opinion on the Russia-Ukraine War (Updated Daily)."
    - Features: Comments, sentiment scores, side (Russia, Ukraine, USA), and timestamps.
    - Preprocessing: Sentiment analysis using VADER and Hugging Face models, data cleaning, and feature engineering.
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