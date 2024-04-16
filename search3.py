import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load the books dataset
books = pd.read_csv('books.csv')

# Function for searching books
def search_books(query):
    # Select specific columns and preprocess the titles for search
    selected_columns = ['book_id', 'title', 'ratings_count', 'image_url']
    df = books[selected_columns]
    df["mod_title"] = df["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["mod_title"] = df["mod_title"].str.lower()
    df["mod_title"] = df["mod_title"].str.replace("\s+", " ", regex=True)
    df = df[df["mod_title"].str.len() > 0]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df["mod_title"])

    # Process query and find similar books
    processed_query = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer.transform([processed_query])
    similarity = cosine_similarity(query_vec, tfidf).flatten()

    # Get top 5 similar books
    k = 5
    indices = np.argpartition(similarity, -k)[-k:]
    results = df.iloc[indices]
    results = results.sort_values("ratings_count", ascending=False)

    return results.head(5)

# Streamlit App
def main():
    st.title("Book Search Engine")
    query = st.text_input("Enter a book title:")
    if st.button("Search"):
        if query:
            st.header("Search Results")
            search_results = search_books(query)
            columns = st.columns(len(search_results))
            for idx, book in enumerate(search_results.itertuples(), start=0):
                with columns[idx % len(search_results)]:
                    st.image(book.image_url, caption=book.title, width=100)
                    st.write(f"Title: {book.title}")
                    st.write(f"Ratings Count: {book.ratings_count}")

if __name__ == "__main__":
    main()
