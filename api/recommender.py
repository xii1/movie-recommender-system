import pandas as pd
from flask import Blueprint, request, jsonify

from __init__ import db, cache
from ml.recommendation import train_rating_model_with_svd, get_n_popular_movies, \
    get_n_rating_movies, predict_rating_with_svd, get_n_recommended_movies_for_user, predict_rating_with_nn, \
    get_n_trending_movies, get_n_similar_movies, calc_tfidf_matrix

recommender = Blueprint('recommender', __name__)

N_TOP_DEFAULT = 10


@recommender.route('/trend/now', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_trending_movies():
    query = {}
    genres = request.args.get('genres')
    top = request.args.get('top')

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return get_n_trending_movies(data, top).to_json(orient='records')


@recommender.route('/trend/popular', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_popular_movies():
    query = {}
    genres = request.args.get('genres')
    top = request.args.get('top')

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return get_n_popular_movies(data, top).to_json(orient='records')


@recommender.route('/trend/rating', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_rating_movies():
    query = {}
    genres = request.args.get('genres')
    top = request.args.get('top')

    if genres is not None:
        query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = list(db.tmdb_movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    return get_n_rating_movies(data, top).to_json(orient='records')


@recommender.route('/similar', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_similar_movies():
    movie = request.args.get('movie')
    top = request.args.get('top')

    if not movie:
        return jsonify({'message': 'Missing movie'})

    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    movies = get_n_similar_movies(movie, top)
    tmdb_movies = {m['original_title']: m
                   for m in list(db.tmdb_movies.find({'original_title': {'$in': movies}},
                                                     {'_id': False, 'id': True, 'original_title': True,
                                                      'genres': True, 'imdb_id': True}))}
    movies = [tmdb_movies[k] for k in movies]

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
    top = request.args.get('top')
    watched = request.args.get('watched')

    genres_query = {}
    if genres is not None:
        genres_query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

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

    return recommended_movies.to_json(orient='records')


# Hybrid recommendation based on the watched movie of a user
@recommender.route('/user/<user_id>/watched', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_top_recommended_movies_with_watched_movie(user_id):
    user_id = int(user_id)
    movie = request.args.get('movie')
    top = request.args.get('top')

    if not movie:
        return jsonify({'message': 'Missing movie'})

    if not top:
        top = N_TOP_DEFAULT
    else:
        top = int(top)

    similar_movies = get_n_similar_movies(movie, 5 * top)
    movies = list(db.movies.find({'title': {'$regex': '|'.join(similar_movies)}}, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    recommended_movies = get_n_recommended_movies_for_user(user_id, top, data)

    # map imdbId by links
    imdb = list(db.links.find({'movieId': {'$in': list(recommended_movies['movieId'])}}))
    imdb = {i['movieId']: i['imdbId'] for i in imdb}

    recommended_movies['imdb_id'] = recommended_movies.apply(lambda m: imdb[m['movieId']], axis=1)

    return recommended_movies.to_json(orient='records')
