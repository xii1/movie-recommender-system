# Movie Recommender System

Movie Recommender System is a REST API application which written by Python (using Flask, UWSGI, MongoDB, Redis, nginx, sklearn).\
Build a movie recommender system utilizing various techniques and serve it via a Restful API. These techniques include: Demographic, content based similarity, and collaborative filtering.\
Objective is to learn recommender systems techniques, and practice embedding and MLOps.

## Getting Started

These instructions will get you building and running the project on your local machine for development and testing purposes. See usage and supported commands for notes on how to use the application.

## Prerequisites

- Python3+
- Docker

## Setup
```bash
./bin/setup
```

## Run for development

- Start mongodb
```bash
./bin/start_db
```
- Start redis cache
```bash
./bin/start_redis
```
- Start localhost server for development (included hot-reload)
```bash
./bin/run
```
- Monitor Redis Cache statistics
```bash
./bin/redis_stats
```

## List of APIs
- Get list of trending movies based on popularity (default top 10)
```
http://localhost:8080/recommender/trend/popular?genres=<string>&top=<number>
```
- Get list of trending movies based on IMDB rating scores (default top 10)
```
http://localhost:8080/recommender/trend/rating?genres=<string>&top=<number>
```
- Get predicted rating of a user for a movie
```
http://localhost:8080/recommender/predict/rating?userId=<number>&movieId=<number>
```
- Get list of recommended movies for a user (default top 10)
```
http://localhost:8080/recommender/user/:user_id?genres=<string>&top=<number>
```

- Train again predicted rating model when add new rating data
```
http://localhost:8080/recommender/train/rating
```
- Clear cache APIs
```
http://localhost:8080/caching/delete/<:key>
http://localhost:8080/clear
```

## Run for production
```bash
./bin/deploy_prod <tag> [scale]
```

## Dataset
- https://www.kaggle.com/juzershakir/tmdb-movies-dataset
- http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

## License
This project is licensed under the MIT License - see the LICENSE file for details