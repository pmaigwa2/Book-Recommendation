import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# Loading trained model and other data
with open('trained_model.pkl', 'rb') as f:
    vectorizer, kmeans, clusters, train_data, tfidf_matrix = pickle.load(f)

# Preprocessing input book name
def preprocess_input(book_name):
    mod_title = re.sub("[^a-zA-Z0-9 ]", "", book_name).lower()
    mod_title = re.sub("\s+", " ", mod_title)
    return mod_title

def find_similar_books(input_book, top_n=10):
    input_mod_title = preprocess_input(input_book)
    input_tfidf = vectorizer.transform([input_mod_title + ' ' + '0']) # Assuming rating as 0 for input book

    # Predict the cluster of the input book
    input_cluster = kmeans.predict(input_tfidf)[0]

    # Filter books belonging to the predicted cluster
    cluster_books = train_data[clusters == input_cluster]

    # Compute cosine similarity between input book and all other books in the same cluster
    similarities = cosine_similarity(input_tfidf, tfidf_matrix[clusters == input_cluster]).flatten()

    # Exclude input book from the similarity calculation
    similarities = similarities[1:]

    # Get indices of top similar books
    top_indices = similarities.argsort()[::-1][:top_n]

    # Get top similar books and their clusters.
    recommended_books = cluster_books.iloc[top_indices][['title', 'cluster']]

    return recommended_books

def get_book_info(recommended_books, books):
    # Filter the 'books' dataset based on recommended book titles.
    recommended_books_info = books[books['title'].isin(recommended_books['title'])]

    # Select and display the relevant information for the recommended books.
    books_info = recommended_books_info[['image_url', 'title', 'authors', 'average_rating', 'original_publication_year']]

    return books_info

# Load books data
books = pd.read_csv('books.csv')

# App layout
st.title('Book Recommendation Model')

st.write('<p style="color:green; line-height: 0.5em;">Enter a book you like to see ten similar books you will love.</p>', unsafe_allow_html=True)
st.markdown('<p style="color:green; line-height: 2.5em;">You can read the recommended books on <a href="https://www.goodreads.com/" style="color:blue;">Goodreads</a>.</p>', unsafe_allow_html=True)
st.write('<p style="color:green; line-height: 2.5em;">Click on a recommended book to view more details about it.</p>', unsafe_allow_html=True)



selected_book = st.selectbox('Enter a book:', books['title'])

def get_recommendations(selected_book):
    recommended_books = find_similar_books(selected_book)
    return recommended_books

# Initialize session state for recommendations
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

# Button
if st.button('Get Recommendations'):
    st.session_state.recommendations = get_recommendations(selected_book)

# Display book titles as buttons( so that users can view more details when they click)
for index, row in st.session_state.recommendations.iterrows():
    if st.button(row['title']):
        # When a book title button is clicked, show the book details
        book_info = get_book_info(st.session_state.recommendations, books)
        for idx, book in book_info.iterrows():
            if book['title'] == row['title']:
                st.image(book['image_url'], width=150)
                st.write(f"Title: {book['title']}")
                st.write(f"Authors: {book['authors']}")
                st.write(f"Average Rating(*/5): {book['average_rating']}")
                st.write(f"Publication Year: {book['original_publication_year']}")
