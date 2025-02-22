# Movie Recommendation System

A simple movie recommendation system built using TF-IDF and cosine similarity. This project leverages a Streamlit demo interface for interactive movie recommendations based on movie plots and genres.

## Features

- **TF-IDF & CountVectorizer:** Uses TF-IDF to vectorize movie plots and CountVectorizer for movie genres.
- **Cosine Similarity:** Computes similarity scores to find the best matches for a given query.
- **Streamlit Interface:** Provides an easy-to-use web interface for movie recommendations.

## Dataset

This project uses the `wiki_movie_plots_deduped.csv` dataset, which contains movie titles, plots, genres, and other metadata. Ensure that this dataset is placed in the project root directory.

Source of the dataset: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots/data

## Setup

### Clone the Repository

```bash
git clone https://github.com/kolanharsha9/lumaa-spring-2025-ai-ml
cd lumaa-spring-2025-ai-ml
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

On Mac/Linux
```bash
source venv/bin/activate  
```
On Windows: 
```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Verify the Dataset

Ensure you have the `wiki_movie_plots_deduped.csv` file in the project root directory.

### Run the Streamlit App

```bash
streamlit run app.py
```

### Interact with the App

A browser window will open displaying the app. Enter your movie query (e.g., "romantic comedy") in the text input field and click the **Recommend** button to see the top 5 movie recommendations.

### Salary Expectations
20$ per hour. 

### Thank you
