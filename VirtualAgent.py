import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Process user queries
queries = ["How can I improve my coding skills?",
           "What are some good programming books to read?"]

# Perform sentiment analysis on queries
sia = SentimentIntensityAnalyzer()
query_sentiments = [sia.polarity_scores(query)["compound"] for query in queries]

# Perform collaborative filtering for recommendations
documents = ["You can practice coding by participating in coding competitions.",
             "Here are some recommended programming books: Clean Code by Robert C. Martin, The Pragmatic Programmer by Andrew Hunt and David Thomas."]

vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)
query_vectors = vectorizer.transform(queries)

similarities = cosine_similarity(query_vectors, document_vectors)

# Get recommendations based on the highest similarity score
recommendations = [documents[similarities[i].argmax()] for i in range(len(queries))]

# Print the results
for i in range(len(queries)):
    print(f"Query: {queries[i]}")
    print(f"Sentiment Score: {query_sentiments[i]}")
    print(f"Recommendation: {recommendations[i]}\n")
