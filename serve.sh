#!/bin/bash
set -e
docker build -t llm-book-blog . && docker run --rm -it -p 8000:8000 -v "$(pwd):/docs" llm-book-blog
