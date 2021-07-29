import os


class Config(object):
    ENV = os.getenv('FLASK_ENV')

    if ENV == 'development':
        MONGO_HOST = 'localhost'
        CACHE_REDIS_HOST = 'localhost'
    else:
        MONGO_HOST = 'mongodb'
        CACHE_REDIS_HOST = 'redis'

    # MongoDB config
    MONGO_PORT = 27017
    MONGO_DBNAME = 'samples'
    MONGO_USERNAME = 'mongo'
    MONGO_PASSWORD = 'mongo'
    MONGO_AUTH_SOURCE = 'admin'
    MONGO_URI = 'mongodb://{}:{}@{}:{}/{}?authSource={}'.format(MONGO_USERNAME, MONGO_PASSWORD,
                                                                MONGO_HOST, MONGO_PORT,
                                                                MONGO_DBNAME, MONGO_AUTH_SOURCE)

    # Redis config
    CACHE_TYPE = 'RedisCache'
    CACHE_REDIS_PORT = 6379
    CACHE_REDIS_DB = 0
    CACHE_DEFAULT_TIMEOUT = 3600
    CACHE_REDIS_URL = 'redis://{}:{}/0'.format(CACHE_REDIS_HOST, CACHE_REDIS_PORT)
