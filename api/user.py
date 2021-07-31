import pandas as pd
from flask import Blueprint, jsonify, request

from __init__ import db, cache

user = Blueprint('user', __name__)


@user.route('/list', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_all_users():
    ids = list(db.ratings.distinct('userId'))
    return jsonify({'ids': ids})


@user.route('/<user_id>/watched', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_watched_movies(user_id):
    user_id = int(user_id)

    genres = request.args.get('genres')

    genres_query = {}
    if genres is not None:
        genres_query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    watched_movies = list(map(lambda m: m['movieId'], list(db.ratings.find({'userId': user_id}))))
    watched_movies_query = {'movieId': {'$in': watched_movies}}

    query = {'$and': [genres_query, watched_movies_query]}
    movies = list(db.movies.find(query, {'_id': False}))
    data = pd.DataFrame(movies)

    if data.size == 0:
        return jsonify([])

    # map imdbId by links
    imdb = list(db.links.find({'movieId': {'$in': list(data['movieId'])}}))
    imdb = {i['movieId']: i['imdbId'] for i in imdb}

    data['imdb_id'] = data.apply(lambda m: imdb[m['movieId']], axis=1)

    return data.to_json(orient='records')
