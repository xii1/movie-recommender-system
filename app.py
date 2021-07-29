from api.caching import caching
from api.recommender import recommender

from __init__ import app

app.register_blueprint(caching, url_prefix='/caching')
app.register_blueprint(recommender, url_prefix='/recommender')

if __name__ == "__main__":
    app.run('localhost', 8080)
