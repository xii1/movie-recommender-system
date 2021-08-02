import pandas as pd
from flask import Blueprint, request, jsonify

from __init__ import db, cache
from ml.recommendation import train_rating_model_with_svd, get_n_popular_movies, \
    get_n_rating_movies, predict_rating_with_svd, get_n_recommended_movies_for_user, predict_rating_with_nn, \
    get_n_trending_movies, get_n_similar_movies, calc_tfidf_matrix

recommender = Blueprint('recommender', __name__)

DEFAULT_N_TOP = 10


@recommender.route('/trend/now', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_trending_movies():
    query = {}
    genres = request.args.get('genres')
    top = int(request.args.get('top', DEFAULT_N_TOP))

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return jsonify(get_n_trending_movies(data, top).to_dict(orient='records'))


@recommender.route('/trend/popular', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_popular_movies():
    query = {}
    genres = request.args.get('genres')
    top = int(request.args.get('top', DEFAULT_N_TOP))

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return jsonify(get_n_popular_movies(data, top).to_dict(orient='records'))


@recommender.route('/trend/rating', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_rating_movies():
    query = {}
    genres = request.args.get('genres')
    top = int(request.args.get('top', DEFAULT_N_TOP))

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return jsonify(get_n_rating_movies(data, top).to_dict(orient='records'))


@recommender.route('/similar', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_similar_movies():
    imdb_id = request.args.get('imdbId')
    top = int(request.args.get('top', DEFAULT_N_TOP))

    if not imdb_id:
        return jsonify({'message': 'Missing imdbId'})

    movie = db.tmdb_movies.find_one({'imdb_id': imdb_id}, {'_id': False, 'original_title': True})
    if movie is None:
        return jsonify({'message': 'Not found movie with imdbId={}'.format(imdb_id)})

    similar_movies = get_n_similar_movies(movie['original_title'], top)
    tmdb_movies = {str(m['original_title']).lower(): m
                   for m in list(db.tmdb_movies.find({'original_title': {'$regex': '|'.join(similar_movies),
                                                                         '$options': 'i'}},
                                                     {'_id': False, 'id': True, 'original_title': True,
                                                      'genres': True, 'imdb_id': True}))}
    movies = [tmdb_movies[k.lower()] for k in similar_movies]

    return jsonify(movies)


@recommender.route('/train/tfidf', methods=['GET'])
def fit_tfidf_matrix():
    movies = list(db.tmdb_movies.find({}, {'_id': False}))
    data = pd.DataFrame(movies)
    calc_tfidf_matrix(data)

    return jsonify({'message': 'Done'})


@recommender.route('/predict/rating', methods=['GET'])
def get_predicted_rating():
    user_id = request.args.get('userId')
    movie_id = request.args.get('movieId')

    if not user_id or not movie_id:
        return jsonify({'message': 'Missing userId or movieId'})
    else:
        user_id = int(user_id)
        movie_id = int(movie_id)

    predicted_rating = dict()
    predicted_rating['SVD'] = predict_rating_with_svd(user_id, movie_id)
    predicted_rating['NN Embedding'] = predict_rating_with_nn([user_id], [movie_id])[0].astype('float64')

    return jsonify({'userId': user_id,
                    'movieId': movie_id,
                    'predicted_rating': predicted_rating
                    })


@recommender.route('/train/rating', methods=['GET'])
def train_rating_model():
    ratings = list(db.ratings.find({}, {'_id': False}))
    data = pd.DataFrame(ratings)
    best_params, best_score = train_rating_model_with_svd(data)

    return jsonify({'message': 'Done', 'SVD': {'best_params': best_params, 'best_score': best_score}})


@recommender.route('/user/<user_id>', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_recommended_movies(user_id):
    user_id = int(user_id)
    genres = request.args.get('genres')
    top = int(request.args.get('top', DEFAULT_N_TOP))
    watched = request.args.get('watched', 'false')

    genres_query = {}
    if genres is not None:
        genres_query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    unwatched_movies_query = {}
    if not watched or watched.lower() in ('false', 'f', '0'):
        watched_movies = list(map(lambda m: m['movieId'], list(db.ratings.find({'userId': user_id}))))
        unwatched_movies_query = {'movieId': {'$nin': watched_movies}}

    query = {'$and': [genres_query, unwatched_movies_query]}
    movies = list(db.movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    recommended_movies = get_n_recommended_movies_for_user(user_id, top, data)

    # map imdbId by links
    imdb = list(db.links.find({'movieId': {'$in': list(recommended_movies['movieId'])}}))
    imdb = {i['movieId']: i['imdbId'] for i in imdb}

    recommended_movies['imdb_id'] = recommended_movies.apply(lambda m: imdb[m['movieId']], axis=1)

    return jsonify(recommended_movies.to_dict(orient='records'))


# Hybrid recommendation based on the watched movie of a user
@recommender.route('/user/<user_id>/watched', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_recommended_movies_with_watched_movie(user_id):
    user_id = int(user_id)
    imdb_id = request.args.get('imdbId')
    top = int(request.args.get('top', DEFAULT_N_TOP))

    if not imdb_id:
        return jsonify({'message': 'Missing imdbId'})

    movie = db.tmdb_movies.find_one({'imdb_id': imdb_id}, {'_id': False, 'original_title': True})
    if movie is None:
        return jsonify({'message': 'Not found movie with imdbId={}'.format(imdb_id)})

    similar_movies = get_n_similar_movies(movie['original_title'], 5 * top)
    movies = list(db.movies.find({'title': {'$regex': '|'.join(similar_movies), '$options': 'i'}}, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    recommended_movies = get_n_recommended_movies_for_user(user_id, top, data)

    # map imdbId by links
    imdb = list(db.links.find({'movieId': {'$in': list(recommended_movies['movieId'])}}))
    imdb = {i['movieId']: i['imdbId'] for i in imdb}

    recommended_movies['imdb_id'] = recommended_movies.apply(lambda m: imdb[m['movieId']], axis=1)

    return jsonify(recommended_movies.to_dict(orient='records'))
