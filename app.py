import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import seaborn as sns



st.title("Data Mining Project by Gagniare Arthur & Aali Andella Mohamed")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    header = st.text_input("Enter header row number", value="0")
    sep = st.text_input("Enter separator (e.g., ',', ';')", value=";")
    df = pd.read_csv(uploaded_file, header=int(header), sep=sep)
    st.write("Data Preview (First and Last 5 rows):")
    st.write(df.head())
    st.write(df.tail())

    st.write("Data Description:")
    st.write(df.describe())

    st.write("Number of missing values per column:")
    st.write(df.isnull().sum())


# Handling missing values
missing_value_option = st.selectbox("Choose method to handle missing values", ["Delete rows", "Delete columns", "Replace with mean", "Replace with median", "Replace with mode", "KNN Imputation"])

if missing_value_option == "Delete rows":
    df_cleaned = df.dropna()
elif missing_value_option == "Delete columns":
    df_cleaned = df.dropna(axis=1)
elif missing_value_option == "Replace with mean":
    df_cleaned = df.fillna(df.mean())
elif missing_value_option == "Replace with median":
    df_cleaned = df.fillna(df.median())
elif missing_value_option == "Replace with mode":
    df_cleaned = df.fillna(df.mode().iloc[0])
# For KNN Imputation, use sklearn's KNNImputer (assuming it's installed)
elif missing_value_option == "KNN Imputation":
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

st.write("Data after handling missing values:")
st.write(df_cleaned)

# Normalizing data
normalization_option = st.selectbox("Choose normalization method", ["Min-Max", "Z-score"])

if normalization_option == "Min-Max":
    df_normalized = (df_cleaned - df_cleaned.min()) / (df_cleaned.max() - df_cleaned.min())
elif normalization_option == "Z-score":
    df_normalized = (df_cleaned - df_cleaned.mean()) / df_cleaned.std()

st.write("Normalized Data:")
st.write(df_normalized)


# Histogram
st.write("Histograms:")
selected_column_hist = st.selectbox("Select column for histogram", df_normalized.columns)
fig, ax = plt.subplots()
ax.hist(df_normalized[selected_column_hist], bins=30)
st.pyplot(fig)

# Box plot
st.write("Box Plots:")
selected_column_box = st.selectbox("Select column for box plot", df_normalized.columns)
fig, ax = plt.subplots()
ax.boxplot(df_normalized[selected_column_box])
st.pyplot(fig)

# Clustering
clustering_option = st.selectbox("Choose clustering algorithm", ["K-Means", "DBSCAN"])

if clustering_option == "K-Means":
    n_clusters = st.slider("Select number of clusters for K-Means", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(df_normalized)
    df_normalized['Cluster'] = clusters
    st.write("Cluster Centers:")
    st.write(kmeans.cluster_centers_)
elif clustering_option == "DBSCAN":
    eps = st.slider("Select epsilon for DBSCAN", 0.1, 10.0, 0.5)
    min_samples = st.slider("Select minimum samples for DBSCAN", 1, 20, 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_normalized)
    df_normalized['Cluster'] = clusters

st.write("Data with Cluster Labels:")
st.write(df_normalized)

# Visualizing clusters
st.write("Cluster Visualization:")
fig, ax = plt.subplots()
scatter = ax.scatter(df_normalized.iloc[:, 0], df_normalized.iloc[:, 1], c=df_normalized['Cluster'], cmap='viridis')
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
st.pyplot(fig)

# Cluster statistics
st.write("Cluster Statistics:")
if clustering_option == "K-Means":
    st.write("Number of data points in each cluster:")
    st.write(df_normalized['Cluster'].value_counts())
    st.write("Cluster Centers:")
    st.write(kmeans.cluster_centers_)
elif clustering_option == "DBSCAN":
    st.write("Number of data points in each cluster:")
    st.write(df_normalized['Cluster'].value_counts())
