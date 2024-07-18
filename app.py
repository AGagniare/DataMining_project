import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler



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

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    
    # sidebar for display graphs
    st.sidebar.title("Data Mining Project")
    show_histogram = st.sidebar.checkbox("Show Histogram")
    show_boxplot = st.sidebar.checkbox("Show Boxplot")

    header = st.text_input("Enter header row number", value="0")
    sep = st.text_input("Enter separator (e.g., ',', ';')", value=",")
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

    if st.button("Handle Missing Values"):
        df_cleaned = handle_missing_values(df, missing_value_option)
        st.write("Data after handling missing values:")
        st.write(df_cleaned)
        df = df_cleaned

    # Data normalization
    normalization_option = st.selectbox("Choose normalization method", ["None", "Min-Max", "Z-score"])
    numeric_columns = df.select_dtypes(include=['number']).columns

    if st.button("Normalize Data"):
        if normalization_option != "None":
            df_normalized = normalize_data(df, normalization_option, numeric_columns)
            st.write("Normalized data:")
            st.write(df_normalized)
            df = df_normalized
        else:
            st.write("No normalization applied.")


    if show_histogram:
        # Histogram
        st.write("Histograms:")
        selected_column_hist = st.selectbox("Select column for histogram", df.columns)
        fig, ax = plt.subplots()
        ax.hist(df[selected_column_hist], bins=30)
        st.pyplot(fig)

    if show_boxplot:
        # Box plot
        st.write("Box Plots:")
        selected_column_box = st.selectbox("Select column for box plot", df.columns)
        fig, ax = plt.subplots()
        ax.boxplot(df[selected_column_box])
        st.pyplot(fig)