import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv("twitter_sentiment.csv")  # Ensure the dataset is in the working directory

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_sentiment(text):
    score = sia.polarity_scores(str(text))  # Convert to string to avoid NaN issues
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['Sentiment'] = df['text'].apply(get_sentiment)

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=df, palette=['red', 'gray', 'green'])
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.title("Sentiment Distribution in Social Media Data")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
