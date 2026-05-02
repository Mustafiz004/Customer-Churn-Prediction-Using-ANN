import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


if "proba" not in st.session_state:
    st.session_state.proba = None

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Load Model & Preprocessors
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_data
def load_preprocessors():
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return label_encoder_gender, onehot_encoder_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_preprocessors()



# -------------------------------
# UI - Title
# -------------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to **leave the bank**.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🧾 Customer Details")

geography = st.sidebar.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox("🧑 Gender", label_encoder_gender.classes_)
age = st.sidebar.slider("🎂 Age", 18, 92, 30)

credit_score = st.sidebar.slider("💳 Credit Score", 300, 900, 600, step=10)

if credit_score < 500:
    st.sidebar.warning("⚠️ Poor")
elif credit_score < 650:
    st.sidebar.info("💡 Fair")
elif credit_score < 750:
    st.sidebar.success("✅ Good")
else:
    st.sidebar.success("💰 Excellent")


balance = st.sidebar.slider("💰 Account Balance", 0, 250000, 50000, step=500)

if balance == 0:
    st.sidebar.warning("⚠️ No balance")
elif balance < 50000:
    st.sidebar.info("💡 Low balance")
elif balance < 150000:
    st.sidebar.success("✅ Moderate balance")
else:
    st.sidebar.success("💰 High-value customer")



estimated_salary = st.sidebar.slider(
    "💼 Estimated Salary",
    0, 200000, 50000, step=500
)

# Feedback
if estimated_salary < 30000:
    st.sidebar.warning("⚠️ Low income range")
elif estimated_salary < 80000:
    st.sidebar.info("💡 Average income level)")
elif estimated_salary < 150000:
    st.sidebar.success("✅ Good income level")
else:
    st.sidebar.success("💰 High income level")


tenure = st.sidebar.selectbox(
    "📆 Years with Bank",
    options=list(range(0, 11)),
    index=3
)
# Number of products (quick selection)
num_of_products = st.sidebar.radio(
    "📦 Number of Products",
    [1, 2, 3, 4],
    horizontal=True
)
has_cr_card = st.sidebar.radio("💳 Has Credit Card?", ["Yes", "No"])
is_active_member = st.sidebar.radio("⚡ Active Member?", ["Yes", "No"])

# Convert categorical inputs
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict Churn"):

    # Prepare data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encoding
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    st.session_state.proba = prediction[0][0]

# -------------------------------
# Prediction Output (Enhanced)
# -------------------------------

# Only show if prediction exists
if st.session_state.proba is not None:


    # -------------------------------
    # Output Section
    # -------------------------------
    st.subheader("📈 Prediction Result")

    # Progress bar
    st.progress(float(st.session_state.proba))

    st.metric(label="Churn Probability", value=f"{st.session_state.proba:.2%}")


    st.subheader("📊 Churn Risk Analysis")

    # -------------------------------
    # Risk categorization
    # -------------------------------
    if st.session_state.proba < 0.3:
        risk_level = "🟢 Low Risk"
    elif st.session_state.proba < 0.7:
        risk_level = "🟠 Medium Risk"
    else:
        risk_level = "🔴 High Risk"

    st.markdown(f"### {risk_level}")

    # -------------------------------
    # Expandable Explanation
    # -------------------------------
    with st.expander("🧠 What does this mean?"):
        st.write(f"""
        - The model estimates a **{st.session_state.proba:.2%} probability** that the customer will churn.
        - This falls under **{risk_level}** category.
        - Predictions are based on customer profile, behavior, and financial data.
        """)

    # -------------------------------
    # Actionable Insights
    # -------------------------------
    st.markdown("### 🎯 Recommended Actions")

    if st.session_state.proba > 0.7:
        st.error("🚨 Immediate attention required!")
        
        st.write("**Suggested Strategy:**")
        st.write("- 📞 Reach out to the customer proactively")
        st.write("- 🎁 Offer personalized retention incentives")
        st.write("- 💬 Assign dedicated support")
        st.write("- 📉 Review dissatisfaction signals")

    elif st.session_state.proba > 0.3:
        st.warning("⚠️ Moderate risk – monitor closely")

        st.write("**Suggested Strategy:**")
        st.write("- 📧 Send engagement emails")
        st.write("- 🎯 Offer targeted promotions")
        st.write("- 🔍 Analyze behavior trends")

    else:
        st.success("✅ Customer is stable")

        st.write("**Suggested Strategy:**")
        st.write("- 🎉 Reward loyalty")
        st.write("- 📈 Upsell premium services")
        st.write("- 🤝 Maintain engagement")

    # -------------------------------
    # Business Insight
    # -------------------------------
    st.markdown("### 💡 Business Insight")

    if st.session_state.proba > 0.5:
        st.write("Losing this customer could impact revenue. Retention efforts are recommended.")
    else:
        st.write("Customer retention likelihood is high. Focus on long-term value growth.")

    # -------------------------------
    # Show last input (optional)
    # -------------------------------
    if "last_input" in st.session_state:
        with st.expander("📋 View Customer Summary"):
            st.write(st.session_state.last_input)

# -------------------------------
# If no prediction yet
# -------------------------------
else:
    st.info("👉 Please click **'Predict Churn'** to see the analysis.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("All Right Reserved ❤️ Md. Mustafizur Rahman")



# for i, layer in enumerate(model.layers):

#     weights = layer.get_weights()

#     with st.expander(f"🔹 Layer {i}: {layer.name}"):

#         if len(weights) == 0:
#             st.write("No weights in this layer.")
#         else:
#             w, b = weights

#             st.write(f"📐 Weights shape: {w.shape}")
#             st.write(f"📐 Bias shape: {b.shape}")

#             st.write("🔢 First Weight:")
#             st.code(str(w.flatten()[:1]))

#             st.write("🔢 First Bias:")
#             st.code(str(b.flatten()[:1]))
