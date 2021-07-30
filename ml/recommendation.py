import dill
import numpy as np
import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import GridSearchCV

QUANTILES_THRESHOLD = 0.95
PREDICTED_RATING_SVD_MODEL_FILE = 'trained_models/recommendation/predicted_rating_svd.pkl'


def get_n_popular_movies(data, n):
    return data.nlargest(n, 'popularity')[['id', 'original_title', 'genres', 'popularity']]


def get_n_rating_movies(data, n):
    m = data['vote_count'].quantile(QUANTILES_THRESHOLD)
    c = data['vote_average'].mean()

    rating_movies = data.copy().loc[data['vote_count'] >= m]
    rating_movies['rating_score'] = rating_movies.apply(lambda movie: calc_weighted_rating(movie, m, c), axis=1)

    return rating_movies.nlargest(n, 'rating_score')[
        ['id', 'original_title', 'genres', 'vote_count', 'vote_average', 'rating_score']]


def calc_weighted_rating(movie, m, c):
    v = movie['vote_count']
    r = movie['vote_average']
    return (v * r + m * c) / (v + m)


def train_rating_model_with_svd(ratings):
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    params = {'n_factors': np.arange(95, 100)}
    gs = GridSearchCV(SVD, param_grid=params, measures=['rmse'], cv=5)
    gs.fit(data)

    svd = gs.best_estimator['rmse']
    svd.fit(data.build_full_trainset())

    with open(PREDICTED_RATING_SVD_MODEL_FILE, "wb") as f:
        dill.dump(svd, f)

    return str(gs.best_params['rmse']), gs.best_score['rmse']


def predict_rating_with_svd(user_id, movie_id):
    model = load_model(PREDICTED_RATING_SVD_MODEL_FILE)
    return model.predict(user_id, movie_id).est


def get_n_recommended_movies_for_user(user_id, n, movies):
    model = load_model(PREDICTED_RATING_SVD_MODEL_FILE)

    # calculate predicted rating of user for all movies
    movies['predicted_rating'] = movies.apply(lambda x: model.predict(user_id, x['movieId']).est, axis=1)

    return movies.nlargest(n, 'predicted_rating')[['movieId', 'title', 'genres', 'predicted_rating']]


def load_model(file_path):
    with open(file_path, "rb") as f:
        return dill.load(f)


# data_df = pd.read_csv('data/movies/tmdb_movies_data.csv')
# movies_df = pd.read_csv('data/movies/movies.csv')
# ratings_df = pd.read_csv('data/movies/ratings.csv')
#
# print(predict_rating_with_svd(1, 1))
# print(get_n_recommended_movies_for_user(2, 5, movies_df))
