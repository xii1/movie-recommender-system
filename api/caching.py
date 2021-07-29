from __init__ import cache

from flask import Blueprint, jsonify


caching = Blueprint('caching', __name__)


@caching.route('/delete/<key>', methods=['GET', 'DELETE'])
def delete_cache(key):
    cache.delete(key)
    return jsonify({'message': 'Delete cache key [{}] successful'.format(key)})


@caching.route('/clear', methods=['GET', 'DELETE'])
def clear_cache():
    cache.clear()
    return jsonify({'message': 'Clear cache successful'})
