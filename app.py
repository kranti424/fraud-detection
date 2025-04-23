import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = pd.read_csv('PS_20174392719_1491204439457_log.csv', nrows=3000)
df['isPayment'] = df['type'].apply(lambda x: 1 if x in ['PAYMENT', 'DEBIT'] else 0)
df['isMovement'] = df['type'].apply(lambda x: 1 if x in ['CASH_OUT', 'TRANSFER'] else 0)
df['accountDiff'] = df['oldbalanceOrg'] - df['oldbalanceDest']
df = df[df['amount'] < 5000000]

X = df[['amount', 'isPayment', 'isMovement', 'accountDiff']].values
y = df['isFraud'].values

# Normalize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# In-built models
log_reg = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
svm = SVC(kernel='linear')

# Train models
log_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Streamlit UI
st.title("Fraud Detection System (Ensemble Voting)")

# Model dictionary
models = {
    "Logistic Regression": log_reg,
    "KNN": knn,
    "Naive Bayes": nb,
    "SVM": svm
}

# Input fields
st.header("Enter Transaction Details")
amount = st.number_input("Transaction Amount", min_value=0.0, value=2000.0)
isPayment = st.selectbox("Is Payment", [1, 0])
isMovement = st.selectbox("Is Movement", [1, 0])
accountDiff = st.number_input("Account Difference", value=100.0)

# Prepare input
input_features = np.array([[amount, isPayment, isMovement, accountDiff]])
input_features = scaler.transform(input_features)

# Predict using all models
if st.button("Make Prediction"):
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(input_features)[0]
            predictions[name] = pred
        except Exception as e:
            st.warning(f"{name} model failed: {e}")
            predictions[name] = None

    # Majority vote
    valid_preds = [p for p in predictions.values() if p is not None]
    if not valid_preds:
        st.error("No valid predictions from models.")
    else:
        vote_counts = Counter(valid_preds)
        majority_vote = vote_counts.most_common(1)[0][0]

        # Display results
        st.subheader("Model Predictions")
        for model_name, pred in predictions.items():
            if pred is not None:
                label = "Fraud" if pred == 1 else "Legit"
                st.write(f"**{model_name}**: {label}")

        st.subheader("Final Decision")
        if majority_vote == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")