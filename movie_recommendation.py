import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """
    Preprocess the dataset: fill missing values and do any necessary cleaning.
    """
    df['Plot'] = df['Plot'].fillna('')
    df['Genre'] = df['Genre'].fillna('')
    # Additional preprocessing (e.g., lowercasing) can be added here.
    return df


def vectorize_features(df):
    """
    Create vector representations for the Plot and Genre fields.
    """
    # TF-IDF for movie plots.
    plot_vectorizer = TfidfVectorizer(stop_words='english')
    plot_tfidf = plot_vectorizer.fit_transform(df['Plot'])

    # CountVectorizer for genres.
    genre_vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    genre_count = genre_vectorizer.fit_transform(df['Genre'])

    # Combine the features.
    combined_features = hstack([plot_tfidf, genre_count])
    return (plot_vectorizer, genre_vectorizer), combined_features


def get_query_vector(query, vectorizers):
    """
    Convert the user query into the same feature space.
    """
    plot_vectorizer, genre_vectorizer = vectorizers
    # Transform the query using the plot vectorizer.
    query_plot_vec = plot_vectorizer.transform([query])
    # Also transform using the genre vectorizer.
    query_genre_vec = genre_vectorizer.transform([query])
    # Combine both vectors.
    query_combined = hstack([query_plot_vec, query_genre_vec])
    return query_combined


def compute_similarity(query_vector, combined_features):
    """
    Compute cosine similarity between the query vector and all movie vectors.
    """
    cosine_sim = cosine_similarity(query_vector, combined_features)
    return cosine_sim


def recommend_movies(query, df, vectorizers, combined_features, top_n=5):
    """
    Recommend the top_n movies most similar to the user's query.
    """
    query_vector = get_query_vector(query, vectorizers)
    cosine_sim = compute_similarity(query_vector, combined_features)

    # Get indices of the top_n most similar movies.
    similar_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    recommendations = df.iloc[similar_indices]
    return recommendations


def main():
    filepath = 'wiki_movie_plots_deduped.csv'
    df = load_data(filepath)

    df = preprocess_data(df)

    vectorizers, combined_features = vectorize_features(df)

    user_query = input("Enter the type of movie you're looking for: ")

    recommendations = recommend_movies(user_query, df, vectorizers, combined_features)
    print("\nTop 5 recommended movies:")
    print(recommendations[['Title', 'Genre', 'Plot']].to_string())


if __name__ == "__main__":
    main()
