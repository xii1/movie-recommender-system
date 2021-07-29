# Movie Recommender System

Movie Recommender System is a REST API application which written by Python (using Flask). Objective is to learn recommender systems techniques, and practice embedding and MLOps.
Build a movie recommender system utilizing various techniques and serve it via a Restful API. These techniques include: Demographic, content based similarity, and collaborative filtering.

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
```bash
./bin/run
```

## Test sample APIs
```bash
http://localhost:8080/sample
http://localhost:8080/sample/hello
http://localhost:8080/visual/sample
http://localhost:8080/visual/water_potability
http://localhost:8080/classifier/dogcat
```

## Run for production
```bash
./bin/deploy_prod <tag> [scale]
```

## License
This project is licensed under the MIT License - see the LICENSE file for details