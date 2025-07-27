import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load models and data
@st.cache_resource
def load_resources():
    kmeans_model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    item_similarity_matrix = pd.read_csv('item_similarity_matrix.csv', index_col=0)
    return kmeans_model, scaler, item_similarity_matrix

kmeans_model, scaler, item_similarity_matrix = load_resources()

cluster_descriptions = {
    0: "Churned/Inactive Customers: These customers have not made a purchase in a long time (High Recency), have a low purchase frequency, and low overall spending.",
    1: "New/Lapsed Customers: These customers have made a purchase recently but have a low purchase frequency and moderate spending. They might be new customers or returning customers who have not yet become frequent buyers.",
    2: "High-Value/Loyal Customers: These customers are highly engaged, with very recent purchases (Low Recency), a high purchase frequency, and the highest overall spending.",
    3: "Frequent/High-Spending Customers: These customers make purchases frequently and have high overall spending, but they have not purchased as recently as the 'High-Value/Loyal Customers'."
}

def get_similar_products(product_title, similarity_matrix):
    """
    Finds the top 5 similar products based on a given product title and similarity matrix.

    Args:
        product_title: The title of the product for which to find similar items.
        similarity_matrix: The DataFrame containing item similarity scores.

    Returns:
        A list of the top 5 most similar product titles or an error message.
    """
    if product_title not in similarity_matrix.index:
        return f"Product '{product_title}' not found in the similarity matrix."

    similar_scores = similarity_matrix[product_title].sort_values(ascending=False)
    # Exclude the product itself and get the top 5
    similar_products = similar_scores[similar_scores.index != product_title].head(5)

    return similar_products.index.tolist()

st.title('Customer Segmentation and Product Recommendation')

st.markdown("""
This application demonstrates customer segmentation using RFM analysis and K-Means clustering, and provides item-based product recommendations.

**RFM Analysis:** RFM (Recency, Frequency, Monetary) analysis is a marketing technique used to quantitatively rank and group customers based on their transaction history.
- **Recency:** How recently a customer has made a purchase.
- **Frequency:** How often a customer makes a purchase.
- **Monetary:** How much money a customer spends.

These three factors provide insights into customer behavior and their potential value to the business.
""")


st.header('Customer Segmentation')
st.markdown("""
This section allows you to predict a customer's segment based on their RFM values using a K-Means clustering model.

**K-Means Clustering:** K-Means is an unsupervised machine learning algorithm used to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype of the cluster. In this application, K-Means clusters customers based on their standardized Recency, Frequency, and Monetary scores.
""")

recency = st.number_input('Enter Recency (days since last purchase):', min_value=0.0)
frequency = st.number_input('Enter Frequency (number of unique invoices):', min_value=0.0)
monetary = st.number_input('Enter Monetary (total spending):', min_value=0.0)

if st.button('Predict Segment'):
    input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    scaled_input = scaler.transform(input_data)
    predicted_cluster = kmeans_model.predict(scaled_input)[0]
    st.subheader('Predicted Customer Segment:')
    st.write(f"Cluster {predicted_cluster}")
    st.write(cluster_descriptions.get(predicted_cluster, "Unknown cluster"))

st.header('Product Recommendation')
st.markdown("""
This section provides product recommendations based on what other customers who bought the same items also purchased. This is an example of item-based collaborative filtering.

**Item-Based Collaborative Filtering:** This recommendation method recommends items to a user based on their similarity to items the user has previously liked or purchased. The similarity between items is typically calculated using metrics like cosine similarity on a user-item matrix, where values represent user interactions (e.g., purchase quantity).

**Cosine Similarity:** Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. In item-based collaborative filtering, it quantifies how similar two items are based on how users have rated or interacted with them. A higher cosine similarity score indicates greater similarity between items.
""")

product_name = st.text_input('Enter a product name to get recommendations:')

if st.button('Get Similar Products'):
    if product_name:
        similar_items = get_similar_products(product_name, item_similarity_matrix)

        if isinstance(similar_items, list):
            st.subheader(f"Top 5 Similar Products to '{product_name}':")
            for item in similar_items:
                st.write(f"- {item}")
        else:
            st.write(similar_items) # Display the error message from the function
    else:
        st.write("Please enter a product name.")
