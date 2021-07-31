from __init__ import app
from api.caching import caching
from api.recommender import recommender
from api.user import user

app.register_blueprint(caching, url_prefix='/caching')
app.register_blueprint(recommender, url_prefix='/recommender')
app.register_blueprint(user, url_prefix='/user')

if __name__ == "__main__":
    app.run('localhost', 8080)
