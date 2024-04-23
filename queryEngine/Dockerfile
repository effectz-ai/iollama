# Stage 1: Build stage
FROM python:3.11.9 AS build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libtesseract-dev \
    tesseract-ocr

# Set the working directory
WORKDIR /app

# Copy only the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=rag_indexer.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV UPLOAD_FOLDER /documents/
ENV HTTP_PORT 7654

WORKDIR /app
COPY ./query_engine.py .
# Specify the command to run your application
EXPOSE 7654
CMD ["flask", "run"]