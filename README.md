# 🏠 House Price Prediction – Machine Learning Project

A complete **end-to-end Machine Learning project** that predicts house prices based on house size using **Linear Regression**.
The project also includes a **deployed Streamlit web application** for real-time predictions.

---

## 🚀 Live Demo (Streamlit App)

👉 **Live Application:**
🔗 [https://housepricepredictionproject-vwd4hylgsfhs94ojfhkutc.streamlit.app/](https://housepricepredictionproject-vwd4hylgsfhs94ojfhkutc.streamlit.app/)

---

## 📂 Project Structure

```
House_Price_Prediction_Project/
│── app.py
│── house_price_model.pkl
│── requirements.txt
│── README.md
```

---

## ✨ Features

### 1️⃣ Data Generation

* Generates a synthetic dataset of **100 house records**
* House sizes range from **500 to 3500 sq ft**
* Prices generated with realistic variation

---

### 2️⃣ Data Preprocessing

* **Missing Value Handling:** Missing prices filled using median
* **Outlier Detection:** IQR (Interquartile Range) method
* **Outlier Removal:** Removes extreme values to improve model quality

---

### 3️⃣ Machine Learning Model

* **Algorithm:** Linear Regression
* **Train–Test Split:** 80% training, 20% testing
* **Feature:** House size (sq ft)
* **Target:** House price

---

### 4️⃣ Model Evaluation

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score (Coefficient of Determination)

---

### 5️⃣ Web Application (Streamlit)

* User-friendly UI
* Input house size
* Instant house price prediction
* Model loaded efficiently using caching
* Deployed on **Streamlit Community Cloud**

---

## 🛠️ Technologies Used

* **Python 3**
* **NumPy** – Numerical computations
* **Pandas** – Data manipulation
* **Matplotlib** – Data visualization
* **Scikit-learn** – ML model & metrics
* **Joblib** – Model serialization
* **Streamlit** – Web application & deployment

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run locally:

```bash
streamlit run app.py
```

### Or use the live app:

🔗 [https://housepricepredictionproject-vwd4hylgsfhs94ojfhkutc.streamlit.app/](https://housepricepredictionproject-vwd4hylgsfhs94ojfhkutc.streamlit.app/)

---

## 📊 Model Output

The model provides:

* Predicted house price
* Learned coefficient (price per sq ft)
* Intercept value
* Evaluation metrics (MSE, RMSE, R²)

---

## 🎯 Learning Outcomes

This project demonstrates:

* End-to-end ML workflow
* Data cleaning & preprocessing
* Handling missing values and outliers
* Model training & evaluation
* Model deployment using Streamlit
* Building ML-powered web applications

---

## 🔮 Future Improvements

* Add more features (location, bedrooms, age, etc.)
* Try advanced models (Ridge, Lasso, Random Forest)
* Improve UI/UX
* Add data upload support
* Store predictions in a database

---

## 👨‍💻 Author

**B Rohit Kumar**
CSE (AI & ML) Student
📌 Machine Learning | Python | Streamlit

**Happy Coding! 🚀**
