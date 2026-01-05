# House Price Prediction - Machine Learning Project

A machine learning project that predicts house prices based on house size using Linear Regression. The project includes data generation, cleaning, model training, and visualization.

## Project Structure

```
ML_Project/
└── House_price_prediction.py
```

## Current Features

### 1. Data Generation
- Generates random dataset of 100 houses
- House sizes range from 500 to 3500 sq ft
- Prices calculated with realistic variation

### 2. Data Preprocessing
- **Missing Value Handling:** Fills missing prices with median values
- **Outlier Detection:** Uses IQR (Interquartile Range) method
- **Outlier Removal:** Removes prices outside the acceptable range

### 3. Machine Learning Model
- **Algorithm:** Linear Regression
- **Train-Test Split:** 80% training, 20% testing
- **Features:** House size (sq ft)
- **Target:** House price ($)

### 4. Model Evaluation
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (coefficient of determination)

### 5. Visualization
- Raw data scatter plot (before cleaning)
- Actual vs Predicted comparison
- Regression line visualization

## Technologies Used

- **Python 3.x**
- **NumPy:** Numerical computations
- **Pandas:** Data manipulation
- **Matplotlib:** Data visualization
- **Scikit-learn:** Machine learning algorithms and metrics

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

Run the prediction model:
```bash
python House_price_prediction.py
```

## Current Model Performance

The model provides:
- Coefficient (slope) indicating price change per sq ft
- Intercept value
- Comparison of actual vs predicted prices for test samples
- Evaluation metrics (MSE, RMSE, R²)

## Future Plans

- **Frontend:** Build a web interface using HTML, CSS, JavaScript or Streamlit
- **Deployment:** Deploy the application to cloud
- **Accuracy Improvement:** Work on improving model accuracy with better features and algorithms

## Learning Outcomes

This project demonstrates:
- End-to-end machine learning workflow
- Data preprocessing and cleaning techniques
- Handling missing values and outliers
- Model training and evaluation
- Data visualization for insights
- Scikit-learn implementation

## Next Steps

1. Experiment with additional features
2. Test different regression algorithms
3. Build the frontend interface
4. Deploy the application to cloud
5. Continuously improve model accuracy

---

**Project:** House Price Prediction  
**Author:** B Rohit Kumar  
**CSE Student**  
**Happy Coding!** 🚀
