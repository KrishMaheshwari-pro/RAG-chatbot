# Makefile for RAG Chatbot project

.PHONY: help install run test clean lint format

help:
	@echo "RAG Chatbot - Available Commands"
	@echo "================================"
	@echo "make install    - Install dependencies"
	@echo "make run        - Run the Streamlit app"
	@echo "make test       - Run test suite"
	@echo "make lint       - Run code linting"
	@echo "make format     - Format code with black"
	@echo "make clean      - Clean cache and build files"

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	pylint *.py

format:
	black *.py tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
