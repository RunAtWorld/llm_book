#!/bin/bash
set -e

docker stop llm-book-blog && \
    docker rm "/llm-book-blog"
sleep 1
docker build -t llm-book-blog . && \
     docker run --rm -d -p 18002:8000 -v "$(pwd):/docs" --name llm-book-blog llm-book-blog
