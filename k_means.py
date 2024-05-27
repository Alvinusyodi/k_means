import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st

# Load the trained KMeans model
customer_data = pd.read_csv("Mall_Customers.csv")
X_train = customer_data.iloc[:, [3, 4]].values
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y_train = kmeans.fit_predict(X_train)

# Title of the app
st.title('Customer Segmentation using K-Means Clustering')

# Form to input new data for prediction
st.subheader('Predict New Data')
annual_income = st.number_input('Annual Income', min_value=0)
spending_score = st.number_input('Spending Score', min_value=0)

if st.button('Predict'):
    new_data = np.array([[annual_income, spending_score]])
    prediction = kmeans.predict(new_data)
    st.write(f'The new data point belongs to cluster {prediction[0] + 1}')
