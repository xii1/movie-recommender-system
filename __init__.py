from flask import Flask
from flask_caching import Cache
from flask_cors import CORS
from flask_pymongo import PyMongo


def create_app():
    flask_app = Flask('app')
    flask_app.config.from_object('config.Config')
    return flask_app


app = create_app()
db = PyMongo(app).db
cache = Cache(app)

CORS(app)
