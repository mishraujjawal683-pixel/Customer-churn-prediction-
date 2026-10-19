
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn AI", page_icon="📊")

st.title("📊 Customer Churn Prediction Using AI")
st.write("Predict whether a customer may leave a service.")

# Create sample data for learning
@st.cache_data
def create_data():
    data = pd.DataFrame({
        "tenure": [2, 5, 12, 24, 36, 48, 3, 18, 30, 60,
                   7, 15, 42, 9, 55, 1, 28, 33, 6, 50],
        "monthly_charges": [90, 85, 70, 60, 55, 50, 95, 75,
                            65, 45, 88, 72, 52, 92, 48, 99,
                            62, 58, 86, 47],
        "contract": ["Month-to-month", "Month-to-month",
                     "Month-to-month", "One year", "One year",
                     "Two year", "Month-to-month", "One year",
                     "Two year", "Two year", "Month-to-month",
                     "One year", "Two year", "Month-to-month",
                     "Two year", "Month-to-month", "One year",
                     "Two year", "Month-to-month", "Two year"],
        "churn": [1, 1, 1, 0, 0, 0, 1, 0, 0, 0,
                  1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    })
    return data

df = create_data()

X = df[["tenure", "monthly_charges", "contract"]]
y = df["churn"]

# Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Convert contract categories into numeric features
preprocessor = ColumnTransformer([
    ("contract", OneHotEncoder(handle_unknown="ignore"),
     ["contract"])
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Evaluation")
st.metric("Test accuracy", f"{accuracy:.0%}")
st.caption(
    "This score uses a tiny synthetic dataset and is not a "
    "reliable estimate of real-world performance."
)

st.subheader("Predict a Customer")

tenure = st.slider("Customer tenure (months)", 0, 72, 6)
charges = st.number_input(
    "Monthly charges", min_value=0.0,
    max_value=1000.0, value=80.0
)
contract = st.selectbox(
    "Contract type",
    ["Month-to-month", "One year", "Two year"]
)

if st.button("Predict Churn"):
    customer = pd.DataFrame({
        "tenure": [tenure],
        "monthly_charges": [charges],
        "contract": [contract]
    })

    result = model.predict(customer)[0]
    probabilities = model.predict_proba(customer)[0]
    churn_probability = probabilities[
        list(model.named_steps["classifier"].classes_).index(1)
    ]

    st.metric("Estimated churn probability",
              f"{churn_probability:.1%}")

    if result == 1:
        st.warning("The model predicts that this customer may churn.")
    else:
        st.success("The model predicts that this customer may stay.")

st.subheader("Sample Customer Data")
st.dataframe(df, use_container_width=True)

