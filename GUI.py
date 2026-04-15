import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_absolute_error, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Background styling
import base64

if "page" not in st.session_state:
    st.session_state.page = "home"

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Add this function to set font color
def set_font_color(color="black"):
    font_style = f"""
    <style>
    .stApp {{
        color: {color};
    }}
    </style>
    """
    st.markdown(font_style, unsafe_allow_html=True)

# Call the function and set the desired font color
set_font_color(color="blue") #B Change "white" to any desired color (e.g., black, blue, etc.)
# Add this function to your code
def set_text_color():
    custom_style = """
    <style>
    .custom-title {
        color: black;
        font-size: 24px;
        font-weight: bold;
    }
    .custom-subtitle {
        color: black;
        font-size: 18px;
    }
    </style>
    """
    st.markdown(custom_style, unsafe_allow_html=True)

# Call the function to set the styling
set_text_color()

# Update your text rendering with custom classes

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_original.csv")
    df['Total sales'] = (df['Unit price'] * df['Quantity']) + df['Tax 5%']
    return df

df = load_data()

# Preprocess dataset
le = LabelEncoder()
df_clean = df.copy()
for col in df_clean.select_dtypes(include='object').columns:
    if col not in ['Invoice ID']:  # Exclude unique identifiers
        df_clean[col] = le.fit_transform(df_clean[col])

# Initialize session state for page navigation  # Default page        



# Welcome Page
if st.session_state.page == "home":
    set_background("D:/DA MINI GUI/s4.jpg")  # Background for the home page
    st.markdown('<p style="color:black; font-size:50px; font-weight:bold;">🧠RevenueVista:</p>', unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;font-size:35px; color: black;'>✨Showcasing the focus on revenue and product popularity✨</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; font-size:35px;color: black;'>CHOOSE AN OPERATION TO PERFORM</h4>", unsafe_allow_html=True)

    # Add buttons at the bottom of the Welcome Page
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Prediction of Gender", key="gender_button"):
            st.session_state.page = "classification_page"
    with col2:
        if st.button("Bulk vs Low item buyers", key="clustering_button"):
            st.session_state.page = "clustering_page"
    with col3:
        if st.button("Highest totalSales Product", key="regression_button"):
            st.session_state.page = "regression_page"
# Classification Page
if st.session_state.page == "classification_page":
    set_background("D:/DA MINI GUI/dim2.jpg")  # Background for the classification page
    # ...existing classification page code...

    # Add "Back to Home Page" button
    if st.button("Back to Home Page"):
        st.session_state.page = "home"

# Regression Page
if st.session_state.page == "regression_page":
    set_background("D:/DA MINI GUI/dim3.jpg")  # Background for the regression page
    # ...existing regression page code...

    # Add "Back to Home Page" button
    if st.button("Back to Home Page"):
        st.session_state.page = "home"

# Clustering Page
if st.session_state.page == "clustering_page":
    set_background("D:/DA MINI GUI/dim4.jpg")  # Background for the clustering page
    # ...existing clustering page code...

    # Add "Back to Home Page" button
    if st.button("Back to Home Page"):
        st.session_state.page = "home"
# Classification Page
# Classification Page
if st.session_state.page == "classification_page":
    
    st.markdown('<p style="color:black; font-size:36px; font-weight:bold;">Gender Purchasing Analysis</p>', unsafe_allow_html=True)

    # Dynamically display column names from the dataset for selection
    
    column_names = df.columns.tolist()  # Get all column names from the dataset
    selected_columns = st.multiselect("Select Columns", options=column_names)

    if st.button("Predict"):
        # Analyze the gender with the highest purchases
        gender_purchases = df.groupby("Gender")["Total sales"].sum().reset_index()
        most_purchasing_gender = gender_purchases.loc[gender_purchases["Total sales"].idxmax()]
        gender_name = most_purchasing_gender["Gender"]
        total_price = most_purchasing_gender["Total sales"]

        # Analyze the most popular product line
        product_purchases = df.groupby("Product line")["Total sales"].sum().reset_index()
        most_purchased_product = product_purchases.loc[product_purchases["Total sales"].idxmax()]
        product_name = most_purchased_product["Product line"]
        product_revenue = most_purchased_product["Total sales"]

        # Display prediction results
        st.markdown('<p style="color:black; font-size:45px; font-weight:bold;">Prediction Results</p>', unsafe_allow_html=True)
        
        st.markdown(f"<p style='color:black; font-size:35px;'>Gender with the Highest Purchases: <b>{gender_name}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:black; font-size:35px;'>Total Price Purchased: <b>₹{total_price:,.2f}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:black; font-size:35px;'>Most Popular Product Line: <b>{product_name}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:black; font-size:35px;'>Total Revenue from {product_name}: <b>₹{product_revenue:,.2f}</b></p>", unsafe_allow_html=True)
        # Display the user's selected columns
        st.markdown('<p style="color:black; font-size:45px; font-weight:bold;">Your Selected Attributes</p>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:black; font-size:20px;'>SELECTED COLUMNS:{selected_columns}</b></p>", unsafe_allow_html=True)
       

# Regression Page
if st.session_state.page == "regression_page":
    st.markdown('<p style="color:black; font-size:45px; font-weight:bold;">📈 Product Popularity Prediction</p>', unsafe_allow_html=True)

    # Dynamically display column names from the dataset for regression
    st.subheader("Choose Columns")
    column_names = df.columns.tolist()  # Get all column names from the dataset
    selected_columns = st.multiselect("Select Feature Columns for Prediction", options=column_names, default=["Unit price", "Quantity"])
    target_column = st.selectbox("Select Target Column for Prediction", options=["Total sales"])

    if st.button("Predict"):
        if selected_columns and target_column:
            X = df_clean[selected_columns]
            y = df_clean[target_column]

            # Perform Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest Regressor
            with st.spinner("Training the regression model..."):
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            # Evaluate Regression
            mae = mean_absolute_error(y_test, preds)

            # Identify the product with the highest revenue
            product_revenue = df.groupby("Product line")["Total sales"].sum().reset_index()
            highest_revenue_product = product_revenue.loc[product_revenue["Total sales"].idxmax()]
            product_name = highest_revenue_product["Product line"]
            highest_revenue = highest_revenue_product["Total sales"]

            # Display results
            st.success("Prediction complete!")
            st.markdown(f"<p style='color:black; font-size:45px;font-weight:bold;'> Predicted Future Trend </b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:black; font-size:30px;'> The Product with the Highest Revenue: : <b>{product_name}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:black; font-size:30px;'>the Highest Revenue : <b>{highest_revenue}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:black; font-size:35px;font-weight:bold;'>Model Evaluation</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:black; font-size:30px;'> Mean Absolute Error: : <b>{mae:.2f}</b></p>", unsafe_allow_html=True)

            # Optional Visualization
            st.markdown(f"<p style='color:black; font-size:35px;font-weight:bold;'>Revenue by Product Line</b></p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=product_revenue, x="Product line", y="Total sales", palette="viridis", ax=ax)
            ax.set_title("Revenue by Product Line")
            ax.set_ylabel("Total Revenue")
            ax.set_xlabel("Product Line")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Please select feature columns and a target column for regression.")


# Clustering Page
if st.session_state.page == "clustering_page":
    st.markdown('<p style="color:black; font-size:45px; font-weight:bold;">🚀Purchase Quantity Analysis</p>', unsafe_allow_html=True)

    # Dynamically display column names from the dataset for clustering
    st.subheader("Choose Columns")
    column_names = df.columns.tolist()  # Get all column names from the dataset
    selected_columns = st.multiselect("Select Columns", options=column_names, default=["Quantity"])  # Default selection is 'Quantity'

    if st.button("Predict"):
        if selected_columns:
            X = df_clean[selected_columns]

            # Perform K-Means Clustering
            k = st.slider("Select Number of attributes", 2, 5, 2)  # Choose number of clusters
            with st.spinner("Performing Prediction..."):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)

            # Add cluster labels to the data
            X["Cluster"] = labels

            # Summarize clustering results
            cluster_means = X.groupby("Cluster").mean()
            max_mean_cluster = cluster_means["Quantity"].idxmax()
            min_mean_cluster = cluster_means["Quantity"].idxmin()

            bulk_items = "Bulk-item buyer" if max_mean_cluster > min_mean_cluster else "Single-item buyer"
            more_quantity_buyer = "Bulk-item buyer" if cluster_means.loc[max_mean_cluster]["Quantity"] > cluster_means.loc[min_mean_cluster]["Quantity"] else "Single-item buyer"

            # Display results
            st.success("Prediction complete!")
        
            st.markdown(f"<p style='color:black; font-size:45px;font-weight:bold;'>Prediction Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:black; font-size:35px;'>The more quantity of product buyers is: {more_quantity_buyer}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:black; font-size:35px;'>Buyer Type with Highest Average Quantity:bulk-items buyer</b></p>", unsafe_allow_html=True)
           
        else:
            st.warning("Please select at least one column for clustering.")
