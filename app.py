import streamlit as st
from movie_recommendation import load_data, preprocess_data, vectorize_features, recommend_movies

@st.cache_data
def load_and_prepare_data():
    '''
    Load and preprocess data
    '''
    filepath = 'wiki_movie_plots_deduped.csv'
    df = load_data(filepath)
    df = preprocess_data(df)

    vectorizers, combined_features = vectorize_features(df)
    return df, vectorizers, combined_features

def main():
    st.title("Movie Recommendation System")
    st.write("Enter the type of movie you're looking for and get the top 5 recommendations.")

    df, vectorizers, combined_features = load_and_prepare_data()

    user_query = st.text_input("Enter your movie query:", "e.g. romantic comedy")

    if st.button("Recommend"):
        if user_query:
            recommendations = recommend_movies(user_query, df, vectorizers, combined_features)
            st.subheader("Top 5 Recommended Movies")
            st.write(recommendations[['Title', 'Genre', 'Plot']])
        else:
            st.error("Please enter a movie query.")

if __name__ == "__main__":
    main()
