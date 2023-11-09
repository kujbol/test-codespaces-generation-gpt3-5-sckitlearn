
import streamlit as st
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Display package versions
st.write(f"scikit-learn version: {sklearn_version}")

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create logistic regression model
lr_model = LogisticRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions on test set
y_pred = lr_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the operations and results
st.write("## scikit-learn Operations and Results")
st.write("### Operation 1: Load Iris dataset")
st.write("Iris dataset is a popular dataset often used for classification tasks.")

st.write("### Operation 2: Split dataset into train and test sets")
st.write("This operation divides the dataset into training and testing subsets.")

st.write("### Operation 3: Create Logistic Regression model")
st.write("Logistic Regression is a popular algorithm for binary classification tasks.")

st.write("### Operation 4: Train the model")
st.write("The model is trained using the training dataset.")

st.write("### Operation 5: Make predictions on test set")
st.write("Using the trained model, predictions are made on the test dataset.")

st.write("### Operation 6: Calculate accuracy")
st.write(f"The accuracy of the Logistic Regression model on the test set is: {accuracy:.2f}")

# Display required dependencies with exact versions
st.write("## Required Dependencies with Exact Versions")
st.write("- Streamlit version: X.X.X")
st.write(f"- scikit-learn version: {sklearn_version}")
