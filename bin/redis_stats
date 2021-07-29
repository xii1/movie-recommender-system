#!/usr/bin/env bash

REDIS_DOCKER_NAME=redis
if [ $# -gt 0 ]
  then
    REDIS_DOCKER_NAME=$1
fi

docker exec -it $REDIS_DOCKER_NAME redis-cli info stats | grep -E '^keyspace|expired_keys|evicted_keys'
