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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.title("Data Mining Project by Gagniare Arthur & Aali Andella Mohamed")

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'normalized_df' not in st.session_state:
    st.session_state.normalized_df = None

def handle_missing_values(df, method):
    if method == "Delete rows":
        df_cleaned = df.dropna()
    elif method == "Delete columns":
        df_cleaned = df.dropna(axis=1)
    elif method == "Replace with mean":
        df_cleaned = df.fillna(df.mean(numeric_only=True))
    elif method == "Replace with median":
        df_cleaned = df.fillna(df.median(numeric_only=True))
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
    missing_value_option = st.selectbox("Choose method to handle missing values", ["None","Delete rows", "Delete columns", "Replace with mean", "Replace with median", "Replace with mode", "KNN Imputation"])

    if missing_value_option != "None":
        st.session_state.cleaned_df = handle_missing_values(df, missing_value_option)
        st.write("Data after handling missing values:")
        st.write(st.session_state.cleaned_df)
        return st.session_state.cleaned_df
    else:
        st.write("No missing value handling applied.")
        return df


def normalize_data_ui(df):
    st.subheader("Data Normalization")
    normalization_option = st.selectbox("Choose normalization method", ["None", "Min-Max", "Z-score"])
    numeric_columns = df.select_dtypes(include=['number']).columns
    if normalization_option != "None":
        st.session_state.normalized_df = normalize_data(df, normalization_option, numeric_columns)
        st.write("Normalized data:")
        st.write(st.session_state.normalized_df)
        return st.session_state.normalized_df
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
            evaluate_clusters(new_df, labels, 'DBSCAN')

def prediction(df, copy_df, target_col):
    st.subheader("Prediction")
    prediction_option = st.selectbox("Choose prediction algorithm", ["Random Forest Classification", "Logistic Regression"])
    feature_cols = st.multiselect("Select feature columns", df.columns)

    if feature_cols and target_col:
        X = df[feature_cols]
        y = copy_df[target_col]

        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=5)

        if prediction_option == "Random Forest Classification":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Prediction results:")
            st.write(pd.DataFrame({"Actual": label_encoder.inverse_transform(y_test), "Predicted": label_encoder.inverse_transform(y_pred)}))
            st.write("Model Accuracy:", accuracy_score(y_test, y_pred))

        elif prediction_option == "Logistic Regression":
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Prediction results:")
            st.write(pd.DataFrame({"Actual": label_encoder.inverse_transform(y_test), "Predicted": label_encoder.inverse_transform(y_pred)}))
            st.write("Model Accuracy:", accuracy_score(y_test, y_pred))

def main():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        num_columns = len(df.columns)
        column_names = [f"Feature{i}" for i in range(num_columns - 1)] + ["Class"]
        df.columns = column_names
        copy_df = df.copy(deep=True)

        # Remove and store target column
        target_col = column_names[-1]
        df = df.drop(target_col, axis=1)

        st.session_state.df = df  # Store original data in session state
        st.session_state.copy_df = copy_df
        data_exploration(copy_df)
        
        # Handling missing values and normalization
        cleaned_df = handle_missing_values_ui(st.session_state.df)
        normalized_df = normalize_data_ui(st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df)
        
        # Store intermediate data states in session state
        if cleaned_df is not None:
            st.session_state.cleaned_df = cleaned_df
        if normalized_df is not None:
            st.session_state.normalized_df = normalized_df

        visualize_data(st.session_state.normalized_df if st.session_state.normalized_df is not None else st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df)

        # Clustering and Prediction
        clustering(st.session_state.normalized_df if st.session_state.normalized_df is not None else st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df)
        prediction(st.session_state.normalized_df if st.session_state.normalized_df is not None else st.session_state.cleaned_df if st.session_state.cleaned_df is not None else df, st.session_state.copy_df, target_col)
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    main()