#!/bin/bash
set -e
docker build -t llm-book-blog . && docker run --rm -it -p 18002:8000 -v "$(pwd):/docs" llm-book-blog
