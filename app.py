import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("🏃‍♂️ Employee Attrition Prediction Dashboard")
st.write("Predict the likelihood of an employee leaving the company based on their demographic and role profile.")

# --- Data Loading ---
@st.cache_data
def load_data():
    # Load the pre-loaded data 
    return pd.read_csv("karmanta_clean.csv")

df = load_data()

# Optional: Show raw data
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df.head())

# --- Model Preparation ---
@st.cache_resource
def train_model(data):
    df_model = data.copy()
    
    # 1. Preprocess target exactly as in ml.ipynb
    df_model["Attrition_flag"] = df_model["Attrition"].map({"Yes": 1, "No": 0})
    X = df_model.drop(columns=["Employee_ID", "Attrition", "Attrition_flag"])
    y = df_model["Attrition_flag"]
    
    # Identify column types
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # 2. Encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale the data (This resolves the LogisticRegression ConvergenceWarning in your notebook)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 5. Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    return lr_model, scaler, X_encoded.columns, cat_cols, num_cols, X

# Initialize model and preprocessing objects
model, scaler, model_columns, categorical_cols, numerical_cols, X_raw = train_model(df)

# --- User Input Sidebar ---
st.sidebar.header("🔍 Employee Parameters")
st.sidebar.write("Adjust the parameters below to predict attrition risk:")

# Capture user inputs dynamically based on the dataset
user_input = {}

# Expose the most impactful features on the sidebar
st.sidebar.subheader("Demographics & Compensation")
user_input["Age"] = st.sidebar.slider("Age", int(X_raw["Age"].min()), int(X_raw["Age"].max()), int(X_raw["Age"].median()))
user_input["Salary"] = st.sidebar.number_input("Salary", int(X_raw["Salary"].min()), int(X_raw["Salary"].max()), int(X_raw["Salary"].median()))
user_input["Distance_From_Home"] = st.sidebar.slider("Distance From Home", int(X_raw["Distance_From_Home"].min()), int(X_raw["Distance_From_Home"].max()), int(X_raw["Distance_From_Home"].median()))

st.sidebar.subheader("Role & Satisfaction")
user_input["Sub-department"] = st.sidebar.selectbox("Sub-department", X_raw["Sub-department"].unique())
user_input["Overtime"] = st.sidebar.selectbox("Overtime", X_raw["Overtime"].unique())
user_input["Business_Travel"] = st.sidebar.selectbox("Business Travel", X_raw["Business_Travel"].unique())
user_input["Job_Satisfaction"] = st.sidebar.slider("Job Satisfaction (1-5)", 1, 5, 3)
user_input["Work_life_balance"] = st.sidebar.slider("Work Life Balance (1-5)", 1, 5, 3)

# Fill unexposed features with default median/mode values so the model has a complete row to predict on
for col in X_raw.columns:
    if col not in user_input:
        if col in categorical_cols:
            user_input[col] = X_raw[col].mode()[0]
        else:
            user_input[col] = X_raw[col].median()

input_df = pd.DataFrame([user_input])

st.subheader("Selected Employee Features")
st.dataframe(input_df[["Age", "Salary", "Distance_From_Home", "Sub-department", "Overtime", "Business_Travel", "Job_Satisfaction", "Work_life_balance"]])

# --- Prediction Logic ---
# 1. Dummy encode the user input
input_encoded = pd.get_dummies(input_df)

# 2. Align columns with training data (fills missing dummy columns safely with 0)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# 3. Scale user features using the pre-fitted scaler
input_scaled = scaler.transform(input_encoded)

# 4. Predict
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.divider()
st.subheader("🔮 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Attrition**")
    else:
        st.success("✅ **Low Risk of Attrition**")

with col2:
    st.metric(label="Attrition Probability", value=f"{prediction_proba[0][1] * 100:.2f}%")