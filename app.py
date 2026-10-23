import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="EM Wave Predictor", page_icon="🔥", layout="wide")

# 2. Load the dataset and the trained model
@st.cache_data
def load_data():
    return pd.read_csv("clean_em_wave_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("em_wave_predictor.pkl")

df = load_data()
model = load_model()

# Header
st.title("🔥 EM Wave Path Prediction Dashboard")
st.markdown("Predicting electromagnetic signal deviation through a turbulent wood-burning thermal plume.")

# Create Tabs for a better UI
tab1, tab2, tab3 = st.tabs(["🎥 Live Frame Analysis", "🎛️ Manual Simulator", "📊 Dataset Overview"])

# --- TAB 1: LIVE FRAME ANALYSIS ---
with tab1:
    st.sidebar.header("Frame Selection")
    row_index = st.sidebar.slider("Select a row from the test data:", 0, len(df)-1, 1500)
    selected_data = df.iloc[row_index]

    st.subheader(f"Analyzing Frame ID: {int(selected_data['frame_id'])}")

    # Display the extracted optical features
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Brightness", f"{selected_data['mean_brightness']:.2f}")
    col2.metric("Thermal Shimmer", f"{selected_data['thermal_shimmer_blur']:.2f}")
    col3.metric("Edge Density", f"{selected_data['refraction_edge_density']:.0f}")

    # Make the Prediction (Passed as a DataFrame to fix the sklearn warning)
    features = pd.DataFrame([selected_data[['mean_brightness', 'thermal_shimmer_blur', 'refraction_edge_density']]])
    prediction = model.predict(features)[0]
    actual = selected_data['target_em_deviation']

    st.markdown("---")
    st.success(f"**Predicted EM Wave Deviation:** {prediction:.6f} units")
    st.info(f"**Actual (Synthesized) Deviation:** {actual:.6f} units")

    # Add a localized trend graph
    st.subheader("Local Distortion Trend (±50 Frames)")
    start_idx = max(0, row_index - 50)
    end_idx = min(len(df), row_index + 50)
    trend_df = df.iloc[start_idx:end_idx]
    st.line_chart(trend_df.set_index('frame_id')[['target_em_deviation']])

# --- TAB 2: MANUAL SIMULATOR ---
with tab2:
    st.subheader("Manual Physics Simulator")
    st.write("Adjust the physical optical metrics manually to see how the Random Forest model reacts in real-time.")
    
    # Sliders based on the min/max values of your actual dataset
    m_bright = st.slider("Mean Brightness (Combustion Intensity)", float(df['mean_brightness'].min()), float(df['mean_brightness'].max()), float(df['mean_brightness'].mean()))
    t_shimmer = st.slider("Thermal Shimmer (Laplacian Variance)", float(df['thermal_shimmer_blur'].min()), float(df['thermal_shimmer_blur'].max()), float(df['thermal_shimmer_blur'].mean()))
    e_density = st.slider("Edge Density (Refractive Boundaries)", float(df['refraction_edge_density'].min()), float(df['refraction_edge_density'].max()), float(df['refraction_edge_density'].mean()))
    
    manual_features = pd.DataFrame([[m_bright, t_shimmer, e_density]], columns=['mean_brightness', 'thermal_shimmer_blur', 'refraction_edge_density'])
    manual_pred = model.predict(manual_features)[0]
    
    st.markdown("---")
    st.warning(f"**Simulated EM Wave Deviation:** {manual_pred:.6f} units")

# --- TAB 3: DATASET OVERVIEW ---
with tab3:
    st.subheader("Dataset Overview")
    st.write(f"Total valid thermal plume frames: {len(df)}")
    st.dataframe(df) 
    
    st.subheader("Overall Target Deviation Trend")
    st.line_chart(df.set_index('frame_id')['target_em_deviation'])