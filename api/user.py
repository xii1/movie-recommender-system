import pandas as pd
from flask import Blueprint, jsonify, request
from flask_pymongo import DESCENDING

from __init__ import db, cache

user = Blueprint('user', __name__)

DEFAULT_PAGE = 1
DEFAULT_NUM_PER_PAGE = 10


@user.route('/list', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_all_users():
    ids = list(db.ratings.distinct('userId'))
    return jsonify({'ids': ids})


@user.route('/<user_id>/watched', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_watched_movies(user_id):
    user_id = int(user_id)
    page = int(request.args.get('page', DEFAULT_PAGE))
    num_per_page = int(request.args.get('numPerPage', DEFAULT_NUM_PER_PAGE))

    genres = request.args.get('genres')

    genres_query = {}
    if genres is not None:
        genres_query = {'genres': {'$regex': '{}'.format(genres), '$options': 'i'}}

    watched_movies = list(map(lambda m: m['movieId'], list(db.ratings.find({'userId': user_id}))))
    watched_movies_query = {'movieId': {'$in': watched_movies}}

    query = {'$and': [genres_query, watched_movies_query]}

    total = db.movies.find(query).count()

    if total == 0:
        return jsonify({'total': 0, 'total_page': 0, 'movies': []})

    if total % num_per_page > 0:
        total_page = int(total / num_per_page) + 1
    else:
        total_page = int(total / num_per_page)

    movies = list(db.movies.find(query, {'_id': False})
                  .sort([('_id', DESCENDING)])
                  .skip(num_per_page * (page - 1))
                  .limit(num_per_page))
    data = pd.DataFrame(movies)

    # map imdbId by links
    imdb = list(db.links.find({'movieId': {'$in': list(data['movieId'])}}))
    imdb = {i['movieId']: i['imdbId'] for i in imdb}

    data['imdb_id'] = data.apply(lambda m: imdb[m['movieId']], axis=1)

    return jsonify({'total': total, 'total_page': total_page, 'page': page, 'movies': data.to_dict(orient='records')})
