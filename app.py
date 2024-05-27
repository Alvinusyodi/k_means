import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import streamlit as st

# Title of the app
st.title('Customer Segmentation using K-Means Clustering')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])

if uploaded_file is not None:
    # Load the data from csv file to a Pandas DataFrame
    customer_data = pd.read_csv(uploaded_file)
    
    # Display first 5 rows of the dataframe
    st.subheader('Dataset')
    st.write(customer_data.head())
    
    # Finding the number of rows and columns
    st.text(f'Dataset contains {customer_data.shape[0]} rows and {customer_data.shape[1]} columns')

    # Getting some informations about the dataset
    st.text('Dataset Information:')
    buffer = customer_data.info(buf=None)
    st.text(buffer)

    # Checking for missing values
    st.subheader('Missing Values:')
    st.write(customer_data.isnull().sum())

    # Extracting the features for clustering
    X = customer_data.iloc[:, [3, 4]].values

    # Finding wcss value for different number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plotting the Elbow Graph
    st.subheader('The Elbow Point Graph')
    plt.figure(figsize=(8, 6))
    sns.set()
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Point Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    # Creating KMeans model with optimal number of clusters
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(X)

    # Plotting all the clusters and their Centroids
    st.subheader('Customer Groups')
    plt.figure(figsize=(8, 8))
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
    plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
    plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
    plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
    plt.title('Customer Groups')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    st.pyplot(plt)
    
    # Form to input new data for prediction
    st.subheader('Predict New Data')
    annual_income = st.number_input('Annual Income', min_value=0)
    spending_score = st.number_input('Spending Score', min_value=0)
    
    if st.button('Predict'):
        new_data = np.array([[annual_income, spending_score]])
        prediction = kmeans.predict(new_data)
        st.write(f'The new data point belongs to cluster {prediction[0] + 1}')
