# 📉 Customer Churn Prediction Using Artificial Neural Network (ANN)

## 📌 Problem Description

Customer churn is one of the most critical challenges faced by businesses in industries such as banking, telecommunications, and subscription-based services. Churn occurs when customers stop using a company's services or switch to a competitor.

Understanding which customers are likely to leave is extremely valuable for organizations because acquiring new customers is often significantly more expensive than retaining existing ones.

This project aims to develop a **machine learning model capable of predicting customer churn** using an **Artificial Neural Network (ANN)** built from scratch. The goal is to identify patterns in customer behavior and predict whether a customer will leave the company or continue using its services.

By accurately predicting churn, businesses can take proactive measures such as offering promotions, improving customer service, or providing personalized offers to retain valuable customers.

This project is intentionally implemented **from a raw foundational level** to clearly demonstrate the **complete machine learning workflow**, including data preprocessing, feature engineering, neural network construction, training, and prediction.

---

# 🧠 Project Overview

This project builds a **binary classification model** using an Artificial Neural Network to predict whether a customer will churn.

The workflow includes:

- Data preprocessing  
- Handling categorical variables  
- Feature scaling  
- Training an Artificial Neural Network  
- Model evaluation  
- Making predictions  

The model learns patterns from historical customer data and predicts the probability of churn.

---

# 📊 Dataset

This project uses the **Bank Customer Churn dataset**, which contains customer information and whether they left the bank.

The dataset includes multiple features related to customer demographics, account information, and engagement levels.

### Dataset Structure

| Feature | Description |
|------|------|
| CreditScore | Customer credit score |
| Geography | Customer's country |
| Gender | Male or Female |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer owns a credit card |
| IsActiveMember | Customer activity status |
| EstimatedSalary | Customer estimated salary |
| Exited | Target variable (1 = churn, 0 = stay) |

### Example Dataset Structure

| CreditScore | Geography | Gender | Age | Balance | NumOfProducts | IsActiveMember | Exited |
|------|------|------|------|------|------|------|------|
| 619 | France | Female | 42 | 0 | 1 | 1 | 1 |
| 608 | Spain | Male | 41 | 83807 | 1 | 1 | 0 |
| 502 | France | Female | 42 | 159660 | 3 | 0 | 1 |

The goal is to train the model to **predict the "Exited" column**.

---

# ⚙️ Methodology

This project demonstrates the **complete machine learning pipeline from scratch**.

The following steps are implemented:

### 1️⃣ Data Preprocessing

The dataset undergoes several preprocessing steps:

- Removing irrelevant columns (CustomerID, RowNumber, Surname)
- Encoding categorical variables
- Converting text data into numerical form
- Feature scaling using **StandardScaler**

Categorical encoding techniques used:

- **Label Encoding**
- **One-Hot Encoding**

---

### 2️⃣ Train-Test Split

The dataset is split into:

- **Training Set:** Used to train the model
- **Test Set:** Used to evaluate the model

Typical split:
80% Training Data
20% Testing Data

This ensures the model is evaluated on unseen data.

---

### 3️⃣ Artificial Neural Network (ANN)

An **Artificial Neural Network** is used to learn complex patterns between customer features and churn behavior.

The network consists of:
Input Layer → Hidden Layer → Hidden Layer → Output Layer


Architecture used in the project:

| Layer | Description |
|------|------|
| Input Layer | Receives processed customer features |
| Hidden Layer 1 | Fully connected layer with ReLU activation |
| Hidden Layer 2 | Fully connected layer with ReLU activation |
| Output Layer | Single neuron with Sigmoid activation |

Activation functions used:

- **ReLU (Rectified Linear Unit)** for hidden layers
- **Sigmoid** for binary classification output

---

### 4️⃣ Model Training

The model is trained using the **Adam optimizer** and **Binary Cross-Entropy loss function**.

Key training parameters:

| Parameter | Value |
|------|------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 100 |
| Batch Size | 32 |

The ANN learns patterns in the training data to predict customer churn probability.

---

### 5️⃣ Model Prediction

After training, the model predicts churn probabilities for new customers.

The output value represents the **probability of churn**:
Output > 0.5 → Customer likely to churn
Output ≤ 0.5 → Customer likely to stay


---

# 📈 Model Evaluation

The model performance is evaluated using a **Confusion Matrix**.

Evaluation metrics include:

- Accuracy
- True Positives
- True Negatives
- False Positives
- False Negatives

These metrics help understand how well the model identifies customers who are likely to leave.

---

# 🛠️ Technologies Used

This project uses the following technologies:

| Technology | Purpose |
|------|------|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Data preprocessing & evaluation |
| TensorFlow / Keras | Neural network implementation |
| Jupyter Notebook | Development environment |

---

# 📂 Project Structure
Customer-Churn-Prediction-Using-ANN
│
├── full_model.ipynb
├── README.md
└── dataset.csv

# 👨‍💻 Author

**Mustafizur Rahman**

Machine Learning & AI Enthusiast
