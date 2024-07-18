import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

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

def prediction(df):
    st.subheader("Prediction")
    prediction_option = st.selectbox("Choose prediction algorithm", ["Linear Regression", "Decision Tree Regression"])

    if prediction_option == "Linear Regression":
        feature_cols = st.multiselect("Select feature columns", df.columns)
        target_col = st.multiselect("Select target column", df.columns)

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
        target_col = st.multiselect("Select target column", df.columns)

        if feature_cols and target_col:
            X = df[feature_cols]
            y = df[target_col]

            model = DecisionTreeRegressor()
            model.fit(X, y)

            st.write("Decision Tree Regression Model:")
            st.write("Feature Importances:", model.feature_importances_)

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    header = st.text_input("Enter header row number", value="0")
    sep = st.text_input("Enter separator (e.g., , or ;)", value=",")
    df = pd.read_csv(uploaded_file, header=int(header), sep=sep)
    copy_df = df.copy(deep=True)

    data_exploration(copy_df)

    df = handle_missing_values_ui(copy_df)

    df = normalize_data_ui(df)

    visualize_data(df)

    clustering(df)

    prediction(df)