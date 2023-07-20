# Step 1: Install required libraries if you haven't already
# pip install nltk
# pip install vaderSentiment

# Step 2: Importing Libraries
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Step 3: Initializing the Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Step 4: Downloading NLTK Resources (One-time setup)
nltk.download('punkt')
nltk.download('vader_lexicon')

# Step 5: Performing Sentiment Analysis
def perform_sentiment_analysis(text):
    scores = analyzer.polarity_scores(text)
    sentiment = ''

    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment

# Step 6: Testing the Sentiment Analysis Function
if __name__ == "__main__":
    reviews = [
        "1-I love the sunny weather! 🌞",
        "2-This movie was terrible and boring. 😔",
        "3-Product X is okay, but it could be better.",
        "4-The restaurant served delicious food!",
        "5-The customer service was excellent.",
        "6-I had a bad experience with their support team.",
        "7-The new software update is fantastic!",
        "8-The book I read was uplifting and inspiring.",
        "9-I can't believe how disappointing the event was.",
        "10-The quality of the product exceeded my expectations.",
        "11-The traffic today was so frustrating!",
        "12-The concert was outstanding and unforgettable.",
        "13-The hotel room was dirty and uncomfortable.",
        "14-The hiking trail offered breathtaking views.",
        "15-The coffee shop had a cozy atmosphere.",
        "16-The application crashed multiple times.",
        "17-The staff at the store was rude and unhelpful.",
        "18-The new feature is a game-changer!",
        "19-The meeting was unproductive and too long.",
        "20-The vacation was relaxing and rejuvenating."
    ]

    for i, review in enumerate(reviews):
        print(f"Review {i+1} Sentiment: {perform_sentiment_analysis(review)}")
