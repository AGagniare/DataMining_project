import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

st.title("Data Mining Project by Gagniare Arthur & Aali Andella Mohamed")

def handle_missing_values(df, method):
    if method == "Delete rows":
        df_cleaned = df.dropna()
    elif method == "Delete columns":
        df_cleaned = df.dropna(axis=1)
    elif method == "Replace with mean":
        df_cleaned = df.fillna(df.mean())
    elif method == "Replace with median":
        df_cleaned = df.fillna(df.median())
    elif method == "Replace with mode":
        df_cleaned = df.fillna(df.mode().iloc[0])
    elif method == "KNN Imputation":
        imputer = KNNImputer(n_neighbors=5)
        df_cleaned = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_cleaned

def normalize_data(df, method, columns=None):
    if columns is None:
        columns = df.columns
    
    if method == "Min-Max":
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
    elif method == "Z-score":
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
    else:
        st.warning("Invalid normalization method selected.")
        return df
    
    return df_normalized

def data_exploration(df):
    st.subheader("Data Exploration")
    st.write("Data Preview (First and Last 5 rows):")
    st.write(df.head())
    st.write(df.tail())

    st.write("Data Description:")
    st.write(df.describe())

    st.write("Number of missing values per column:")
    st.write(df.isnull().sum())

def handle_missing_values_ui(df):
    st.subheader("Missing Value Handling")
    missing_value_option = st.selectbox("Choose method to handle missing values", ["Delete rows", "Delete columns", "Replace with mean", "Replace with median", "Replace with mode", "KNN Imputation"])

    if st.button("Handle Missing Values"):
        df_cleaned = handle_missing_values(df, missing_value_option)
        st.write("Data after handling missing values:")
        st.write(df_cleaned)
        return df_cleaned
    return df

def normalize_data_ui(df):
    st.subheader("Data Normalization")
    normalization_option = st.selectbox("Choose normalization method", ["None", "Min-Max", "Z-score"])
    numeric_columns = df.select_dtypes(include=['number']).columns

    if st.button("Normalize Data"):
        if normalization_option != "None":
            df_normalized = normalize_data(df, normalization_option, numeric_columns)
            st.write("Normalized data:")
            st.write(df_normalized)
            return df_normalized
        else:
            st.write("No normalization applied.")
    return df

def visualize_data(df):
    st.subheader("Data Visualization")
    st.sidebar.title("Data Mining Project")
    show_histogram = st.sidebar.checkbox("Show Histogram")
    show_boxplot = st.sidebar.checkbox("Show Boxplot")

    if show_histogram:
        st.write("Histograms:")
        selected_column_hist = st.selectbox("Select column for histogram", df.columns)
        fig, ax = plt.subplots()
        ax.hist(df[selected_column_hist], bins=30)
        st.pyplot(fig)

    if show_boxplot:
        st.write("Box Plots:")
        selected_column_box = st.selectbox("Select column for box plot", df.columns)
        fig, ax = plt.subplots()
        ax.boxplot(df[selected_column_box])
        st.pyplot(fig)

def evaluate_clusters(df, labels, clustering_method, kmeans=None):
    st.subheader("Cluster Evaluation")

    # Visualization of clusters
    if len(df.columns) > 1:
        col1 = st.selectbox("Select X-axis column for cluster visualization", df.columns)
        col2 = st.selectbox("Select Y-axis column for cluster visualization", df.columns)
        pca = PCA(n_components=2)
        df_pca = pd.DataFrame(pca.fit_transform(df), columns=['PCA1', 'PCA2'])

        fig, ax = plt.subplots()
        scatter = ax.scatter(df_pca['PCA1'], df_pca['PCA2'], c=labels, cmap='viridis')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        st.pyplot(fig)

    # Cluster statistics
    st.write("Cluster Statistics:")
    unique_labels = set(labels)
    for label in unique_labels:
        st.write(f"Cluster {label}:")
        st.write(f"Number of data points: {sum(labels == label)}")
        if clustering_method == 'K-Means':
            cluster_center = kmeans.cluster_centers_[label]
            st.write(f"Cluster center: {cluster_center}")
        elif clustering_method == 'DBSCAN':
            cluster_data = df_pca[labels == label]
            cluster_center = cluster_data.mean(axis=0)
            cluster_density = sum(labels == label) / (df_pca.loc[labels == label].apply(lambda x: np.linalg.norm(x - cluster_center), axis=1).mean())
            st.write(f"Cluster density: {cluster_density}")

def plot_clusters(df, labels, n_clusters):
    if len(df.columns) >= 2:
        col1 = st.selectbox("Select X-axis column", df.columns)
        col2 = st.selectbox("Select Y-axis column", df.columns)

        fig, ax = plt.subplots()
        for cluster in range(n_clusters if n_clusters != -1 else max(labels) + 1):
            cluster_data = df.loc[labels == cluster, [col1, col2]]
            ax.scatter(cluster_data[col1], cluster_data[col2], label=f"Cluster {cluster}")
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough columns for 2D visualization. Minimum 2 columns required.")

# Updated clustering function with evaluation
def clustering(df):
    st.subheader("Clustering")
    show_cluster_plot = st.sidebar.checkbox("Show Cluster Plot")
    clustering_option = st.selectbox("Choose clustering algorithm", ["K-Means", "DBSCAN"])

    new_df = df.copy(deep=True)

    if clustering_option == "K-Means":
        n_clusters = st.slider("Select number of clusters", 2, 10, 5)
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(new_df)
        new_df['Cluster'] = labels
        st.write("Clustering results:")
        st.write(new_df)
        if show_cluster_plot:
            plot_clusters(new_df, labels, n_clusters)
        evaluate_clusters(new_df, labels, 'K-Means', kmeans)

    elif clustering_option == "DBSCAN":
        eps = st.slider("Select epsilon value", 0.1, 10.0, 3.5)
        min_samples = st.slider("Select minimum samples", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(new_df)
        new_df['Cluster'] = labels
        st.write("Clustering results:")
        st.write(new_df)
        if show_cluster_plot:
            plot_clusters(new_df, labels, -1)
        evaluate_clusters(new_df, labels, 'DBSCAN')

def prediction(df):
    st.subheader("Prediction")
    prediction_option = st.selectbox("Choose prediction algorithm", ["Linear Regression", "Decision Tree Regression", "Random Forest Classification"])

    if prediction_option == "Linear Regression":
        feature_cols = st.multiselect("Select feature columns", df.columns)
        target_col = st.selectbox("Select target column", df.columns)

        if feature_cols and target_col:
            X = df[feature_cols]
            y = df[target_col]

            model = LinearRegression()
            model.fit(X, y)

            st.write("Linear Regression Model:")
            st.write("Coefficients:", model.coef_)
            st.write("Intercept:", model.intercept_)

    elif prediction_option == "Decision Tree Regression":
        feature_cols = st.multiselect("Select feature columns", df.columns)
        target_col = st.selectbox("Select target column", df.columns)

        if feature_cols and target_col:
            X = df[feature_cols]
            y = df[target_col]

            model = DecisionTreeRegressor()
            model.fit(X, y)

            st.write("Decision Tree Regression Model:")
            st.write("Feature Importances:", model.feature_importances_)

    elif prediction_option == "Random Forest Classification":
        feature_cols = st.multiselect("Select feature columns", df.columns)
        target_col = st.selectbox("Select target column", df.columns)

        if feature_cols and target_col:
            X = df[feature_cols]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write("Random Forest Classification Model:")
            st.write("Accuracy:", accuracy)

def determine_optimal_clusters(df):
    st.subheader("Optimal Number of Clusters")
    max_clusters = st.slider("Select maximum number of clusters", 2, 20, 10)

    ch_scores = []
    db_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(df)
        ch_scores.append(calinski_harabasz_score(df, labels))
        db_scores.append(davies_bouldin_score(df, labels))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(range(2, max_clusters + 1), ch_scores, marker='o')
    ax[0].set_title('Calinski-Harabasz Index')
    ax[0].set_xlabel('Number of Clusters')
    ax[0].set_ylabel('Score')

    ax[1].plot(range(2, max_clusters + 1), db_scores, marker='o')
    ax[1].set_title('Davies-Bouldin Index')
    ax[1].set_xlabel('Number of Clusters')
    ax[1].set_ylabel('Score')

    st.pyplot(fig)

    st.write("Optimal number of clusters based on Calinski-Harabasz Index:", ch_scores.index(max(ch_scores)) + 2)
    st.write("Optimal number of clusters based on Davies-Bouldin Index:", db_scores.index(min(db_scores)) + 2)

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    header = st.text_input("Enter header row number", value="0")
    sep = st.text_input("Enter separator (e.g., , or ;)", value=",")
    df = pd.read_csv(uploaded_file, header=None, sep=sep)
    
    # Rename columns
    num_columns = len(df.columns)
    column_names = [f"Feature{i}" for i in range(num_columns - 1)] + ["Class"]
    df.columns = column_names
    
    copy_df = df.copy(deep=True)

    data_exploration(copy_df)

    df = handle_missing_values_ui(copy_df)

    df = normalize_data_ui(df)

    visualize_data(df)

    # clustering(df)

    # determine_optimal_clusters(df)

    prediction(df)