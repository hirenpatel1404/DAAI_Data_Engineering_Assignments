import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# I load the Iris dataset that comes with scikit-learn
iris = load_iris()
X = iris.data
y = iris.target

# I split the data: 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# I create the logistic regression model and train it
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# I check how well the model does on the test set
y_pred = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 3))

# I save the trained model to a file so my Flask app can load it later
with open("logistic_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as logistic_model.pkl")