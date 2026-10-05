# I start from an official Python 3.12 image, the same Python version I used to train the model
FROM python:3.12-slim

# I set the working folder inside the container
WORKDIR /app

# I copy the list of libraries first
COPY requirements.txt .

# I install the libraries listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# I copy my Flask app and my saved model into the container
COPY app.py logistic_model.pkl ./

# I note that the app listens on port 5000
EXPOSE 5000

# I start the Flask app when the container starts
CMD ["python", "app.py"]
