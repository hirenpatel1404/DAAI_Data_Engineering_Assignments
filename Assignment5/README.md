# Assignment 5: Deploying a Logistic Regression Model with Flask and Docker

## Overview

In this assignment I trained a logistic regression model on the Iris dataset, saved it with pickle, served it through a Flask API, and packaged the whole application in a Docker container. The API has one endpoint, /predict, which accepts a POST request with four flower measurements and returns the predicted species.

## Files

- `logistic_model.py` trains the model and saves it as `logistic_model.pkl`.
- `logistic_model.pkl` is the saved model.
- `app.py` is the Flask application that loads the model and exposes the /predict endpoint.
- `requirements.txt` lists the libraries and their exact versions.
- `Dockerfile` contains the instructions to build the Docker image.
- `.dockerignore` lists the folders Docker should skip when building.

## Step 1: Train and save the model

```bash
python logistic_model.py
```

Output:

```
Test accuracy: 0.967
Model trained and saved as logistic_model.pkl
```

I split the data into 80% for training and 20% for testing, with a stratified split. The model predicted 29 of the 30 test flowers correctly.

## Step 2: Build the Docker image

```bash
docker build -t iris-flask-api .
```

## Step 3: Run the container on port 5000

```bash
docker run -d -p 5000:5000 --name iris-api iris-flask-api
```

Output of `docker ps`:

```
CONTAINER ID   IMAGE            COMMAND           STATUS                  PORTS                    NAMES
9f6fddf26265   iris-flask-api   "python app.py"   Up Less than a second   0.0.0.0:5000->5000/tcp   iris-api
```

## Step 4: Query the model

Request 1:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Response:

```
{"prediction":0,"species":"setosa"}
```

Request 2:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 1.8}'
```

Response:

```
{"prediction":2,"species":"virginica"}
```

Container log, showing both requests answered with status 200:

```
172.17.0.1 - - [05/Oct/2026 01:16:12] "POST /predict HTTP/1.1" 200 -
172.17.0.1 - - [05/Oct/2026 01:16:48] "POST /predict HTTP/1.1" 200 -
```

## Error handling

Flask does not check the input automatically, so I added my own checks. If a measurement is missing, the API replies with status 400 and names the missing fields:

```
{"error":"Missing fields: petal_length, petal_width"}
```

## What I learned

The API gave the same prediction when I ran it directly on my computer and when I ran it inside the container. This is the main purpose of Docker: the application is packed together with its Python version and its libraries, so it behaves the same on any machine.

A pickle file only loads safely with the same library version that created it. For this reason I trained the model with Python 3.12 and scikit-learn 1.5.2, wrote these versions in requirements.txt, and used the python:3.12-slim image, so the container has exactly the same versions as my computer.

I also learned the difference between an image and a container. The image is the packed application stored on disk, and the container is a running copy of it. The option -p 5000:5000 connects port 5000 on my computer to port 5000 inside the container, and host 0.0.0.0 in the Flask app allows requests from outside the container to reach it.