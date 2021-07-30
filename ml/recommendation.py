import dill
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from surprise import SVD, Reader, Dataset
from surprise.model_selection import GridSearchCV
from tensorflow.keras import layers, activations, models, optimizers, losses

PREDICTED_RATING_SVD_MODEL_FILE = 'trained_models/recommendation/predicted_rating_svd.pkl'
QUANTILES_THRESHOLD = 0.95

PREDICTED_RATING_NN_WITH_EMBEDDING_MODEL = 'trained_models/recommendation/predicted_rating_nn_model'
PREDICTED_RATING_NN_WITH_EMBEDDING_RATING_SCALER_FILE = 'trained_models/recommendation/predicted_rating_nn_rating_scaler.pkl'
PREDICTED_RATING_NN_WITH_EMBEDDING_USER_ENCODER_FILE = 'trained_models/recommendation/predicted_rating_nn_user_encoder.pkl'
PREDICTED_RATING_NN_WITH_EMBEDDING_MOVIE_ENCODER_FILE = 'trained_models/recommendation/predicted_rating_nn_movie_encoder.pkl'
N_FACTORS = 10


def get_n_popular_movies(data, n):
    return data.nlargest(n, 'popularity')[['id', 'original_title', 'genres', 'popularity', 'imdb_id']]


def get_n_rating_movies(data, n):
    m = data['vote_count'].quantile(QUANTILES_THRESHOLD)
    c = data['vote_average'].mean()

    rating_movies = data.copy().loc[data['vote_count'] >= m]
    rating_movies['rating_score'] = rating_movies.apply(lambda movie: calc_weighted_rating(movie, m, c), axis=1)

    return rating_movies.nlargest(n, 'rating_score')[
        ['id', 'original_title', 'genres', 'vote_count', 'vote_average', 'rating_score', 'imdb_id']]


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

    save_obj(svd, PREDICTED_RATING_SVD_MODEL_FILE)  # save model to file

    return str(gs.best_params['rmse']), gs.best_score['rmse']


def predict_rating_with_svd(user_id, movie_id):
    model = load_obj(PREDICTED_RATING_SVD_MODEL_FILE)  # load model from file
    return model.predict(user_id, movie_id).est


def train_rating_model_with_neural_network(ratings):
    ratings = ratings.drop(columns=['timestamp'])

    num_users = len(ratings['userId'].unique())  # calc number of users
    num_movies = len(ratings['movieId'].unique())  # calc number of movies

    # normalize data
    mm_scaler = MinMaxScaler()
    ratings[['rating']] = mm_scaler.fit_transform(ratings[['rating']])

    users = ratings[['userId']].values
    movies = ratings[['movieId']].values
    ratings = ratings[['rating']].values.reshape(-1)

    # encode users
    user_encoder = LabelEncoder()
    users = user_encoder.fit_transform(users)

    # encode movies
    movie_encoder = LabelEncoder()
    movies = movie_encoder.fit_transform(movies)

    # define embedding layer for user
    user_input = layers.Input(shape=(1,))
    user_embed = layers.Embedding(num_users, N_FACTORS)(user_input)
    user_vector = layers.Flatten()(user_embed)

    # define embedding layer for movie
    movie_input = layers.Input(shape=(1,))
    movie_embed = layers.Embedding(num_movies, N_FACTORS)(movie_input)
    movie_vector = layers.Flatten()(movie_embed)

    # merge features
    merge = layers.concatenate([user_vector, movie_vector])
    layer = layers.Dropout(0.5)(merge)

# add fully connected layers with dropout
    layer = layers.Dense(32, activation=activations.relu)(layer)
    layer = layers.Dropout(0.5)(layer)
    layer = layers.Dense(16, activation=activations.relu)(layer)
    layer = layers.Dropout(0.5)(layer)
    output = layers.Dense(1, activation=activations.sigmoid)(layer)

    # create model
    model = models.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(), loss=losses.mean_squared_error)

    # train model
    history = model.fit([users, movies], ratings, validation_split=0.2, batch_size=32, epochs=20, verbose=1)
    model.summary()

    # save all to file
    model.save(PREDICTED_RATING_NN_WITH_EMBEDDING_MODEL)
    save_obj(mm_scaler, PREDICTED_RATING_NN_WITH_EMBEDDING_RATING_SCALER_FILE)
    save_obj(user_encoder, PREDICTED_RATING_NN_WITH_EMBEDDING_USER_ENCODER_FILE)
    save_obj(movie_encoder, PREDICTED_RATING_NN_WITH_EMBEDDING_MOVIE_ENCODER_FILE)

    # visualization train loss / validate loss
    visualize([{"train": history.history["loss"], "validate": history.history["val_loss"]}], ["Model Trained"], ["epoch"], ["loss"])


def predict_rating_with_nn(user_ids, movie_ids):
    # load all from file
    model = models.load_model(PREDICTED_RATING_NN_WITH_EMBEDDING_MODEL)
    mm_scaler = load_obj(PREDICTED_RATING_NN_WITH_EMBEDDING_RATING_SCALER_FILE)
    user_encoder = load_obj(PREDICTED_RATING_NN_WITH_EMBEDDING_USER_ENCODER_FILE)
    movie_encoder = load_obj(PREDICTED_RATING_NN_WITH_EMBEDDING_MOVIE_ENCODER_FILE)

    rating = model.predict([user_encoder.transform(user_ids), movie_encoder.transform(movie_ids)])

    return mm_scaler.inverse_transform(rating).reshape(-1)


def get_n_recommended_movies_for_user(user_id, n, movies):
    model = load_obj(PREDICTED_RATING_SVD_MODEL_FILE)

    # calculate predicted rating of user for all movies
    movies['predicted_rating'] = movies.apply(lambda x: model.predict(user_id, x['movieId']).est, axis=1)

    return movies.nlargest(n, 'predicted_rating')[['movieId', 'title', 'genres', 'predicted_rating']]


def visualize(data, titles, xlabels, ylabels):
    fig, axes = plt.subplots(len(titles), squeeze=False)
    fig.suptitle('Visualization', fontsize=16)

    for i in range(len(titles)):
        axes[i, 0].set_title(titles[i])
        axes[i, 0].set_xlabel(xlabels[i])
        axes[i, 0].set_ylabel(ylabels[i])

        for s in data[i].keys():
            axes[i, 0].plot(data[i][s], label=s)

        axes[i, 0].legend(loc="best")

    plt.grid()
    plt.tight_layout()
    plt.show()


def save_obj(obj, file_path):
    with open(file_path, "wb") as f:
        dill.dump(obj, f)


def load_obj(file_path):
    with open(file_path, "rb") as f:
        return dill.load(f)


# data_df = pd.read_csv('data/movies/tmdb_movies_data.csv')
# movies_df = pd.read_csv('data/movies/movies.csv')
# ratings_df = pd.read_csv('data/movies/ratings.csv')
#
# print(predict_rating_with_svd(1, 47))
# print(get_n_recommended_movies_for_user(2, 5, movies_df))
#
# train_rating_model_with_neural_network(ratings_df)
# print(predict_rating_with_nn([1, 1], [1, 47]))
