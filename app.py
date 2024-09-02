# Import necessary libraries
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
def load_model():
    with open('LogisticR.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Initialize the model
model = load_model()

# Streamlit app title
st.title("Breast Cancer Prediction App")

# Sidebar inputs for user data
st.sidebar.header("Input Features")

# Function to get user inputs from sidebar
def user_input_features():
    # Input sliders for all 30 features in the dataset
    mean_radius = st.sidebar.slider("Mean Radius", 6.0, 30.0, 14.0)
    mean_texture = st.sidebar.slider("Mean Texture", 9.0, 40.0, 20.0)
    mean_perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, 90.0)
    mean_area = st.sidebar.slider("Mean Area", 140.0, 2500.0, 600.0)
    mean_smoothness = st.sidebar.slider("Mean Smoothness", 0.05, 0.2, 0.1)
    
    mean_compactness = st.sidebar.slider("Mean Compactness", 0.01, 0.35, 0.1)
    mean_concavity = st.sidebar.slider("Mean Concavity", 0.0, 0.45, 0.1)
    mean_concave_points = st.sidebar.slider("Mean Concave Points", 0.0, 0.2, 0.05)
    mean_symmetry = st.sidebar.slider("Mean Symmetry", 0.1, 0.4, 0.2)
    mean_fractal_dimension = st.sidebar.slider("Mean Fractal Dimension", 0.04, 0.1, 0.06)
    
    radius_error = st.sidebar.slider("Radius Error", 0.1, 3.0, 0.5)
    texture_error = st.sidebar.slider("Texture Error", 0.3, 5.0, 1.0)
    perimeter_error = st.sidebar.slider("Perimeter Error", 0.7, 22.0, 5.0)
    area_error = st.sidebar.slider("Area Error", 6.0, 550.0, 40.0)
    smoothness_error = st.sidebar.slider("Smoothness Error", 0.002, 0.03, 0.01)
    
    compactness_error = st.sidebar.slider("Compactness Error", 0.002, 0.1, 0.03)
    concavity_error = st.sidebar.slider("Concavity Error", 0.0, 0.4, 0.02)
    concave_points_error = st.sidebar.slider("Concave Points Error", 0.0, 0.05, 0.01)
    symmetry_error = st.sidebar.slider("Symmetry Error", 0.007, 0.08, 0.02)
    fractal_dimension_error = st.sidebar.slider("Fractal Dimension Error", 0.001, 0.03, 0.007)
    
    worst_radius = st.sidebar.slider("Worst Radius", 7.0, 40.0, 16.0)
    worst_texture = st.sidebar.slider("Worst Texture", 12.0, 50.0, 25.0)
    worst_perimeter = st.sidebar.slider("Worst Perimeter", 50.0, 250.0, 100.0)
    worst_area = st.sidebar.slider("Worst Area", 185.0, 4254.0, 880.0)
    worst_smoothness = st.sidebar.slider("Worst Smoothness", 0.07, 0.25, 0.15)
    
    worst_compactness = st.sidebar.slider("Worst Compactness", 0.02, 1.0, 0.25)
    worst_concavity = st.sidebar.slider("Worst Concavity", 0.0, 1.25, 0.27)
    worst_concave_points = st.sidebar.slider("Worst Concave Points", 0.0, 0.3, 0.1)
    worst_symmetry = st.sidebar.slider("Worst Symmetry", 0.1, 0.7, 0.3)
    worst_fractal_dimension = st.sidebar.slider("Worst Fractal Dimension", 0.05, 0.2, 0.1)
    
    # Combine all input features into a single array
    features = np.array([
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
        radius_error, texture_error, perimeter_error, area_error, smoothness_error,
        compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
        worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
    ])
    return features

# Display user input
input_features = user_input_features()
st.subheader("User Input Features")
st.write({
    "Mean Radius": input_features[0],
    "Mean Texture": input_features[1],
    "Mean Perimeter": input_features[2],
    "Mean Area": input_features[3],
    "Mean Smoothness": input_features[4],
    # Display other features as needed
})

# Scaling inputs before prediction
scaler = StandardScaler()
input_features_scaled = scaler.fit_transform([input_features])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_features_scaled)
    if prediction[0] == 1:
        st.write("Prediction: The tumor is **malignant**.")
    else:
        st.write("Prediction: The tumor is **benign**.")
