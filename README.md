#  Factory Reallocation & Shipping Optimization System

#Project Overview
This project is a Machine Learning-based system that helps improve factory assignment and shipping performance for Nassau Candy Distributor.

It predicts shipping time (lead time) and suggests the best factory/region to improve delivery speed and profit.

# Problem Statement
Currently, the company uses fixed rules to assign products to factories. This causes:

- Slow shipping time  
- Poor delivery routes  
- Loss of profit due to wrong decisions  

This project solves these problems using data and Machine Learning.

# Project Objectives
- Predict shipping time using Machine Learning  
- Analyze important business KPIs  
- Suggest the best factory/region  
- Improve delivery speed and profit  

#Dataset Information
The dataset includes:

- Order ID  
- Order Date  
- Ship Date  
- Product Name  
- Region  
- Ship Mode  
- Sales  
- Units  
- Cost  
- Gross Profit  

# Technologies Used
- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

#Machine Learning Model
We used a **Gradient Boosting Regressor** model.

#Input Features:
- Sales  
- Units  
- Cost  
- Gross Profit  

This model helps to predict shipping lead time.

#Key Features
- KPI Dashboard (Sales, Profit, Orders, Lead Time)  
- Lead Time Prediction System  
- Factory/Region Recommendation System  
- Simple Data Visualizations  
- Filters (Product, Region, Ship Mode)  

#How Optimization Works
The system selects the best factory/region based on:

- High profit  
- Fast delivery time  

It tries to balance both profit and speed to give the best recommendation.

# How to Run This Project Locally
```bash
pip install -r requirements.txt
streamlit run app.py
