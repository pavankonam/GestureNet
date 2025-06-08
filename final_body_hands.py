import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Gesture Net",
    page_icon="🧠👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-section {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .gesture-range {
        background-color: #07090a;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        border-left: 3px solid #1f77b4;
    }
    .model-1-section {
        border-color: #d32f2f;
        background-color: #fdf2f2;
    }
    .model-2-section {
        border-color: #388e3c;
        background-color: #f2fdf2;
    }
    .file-upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
    .merge-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_models():
    """Load both trained gesture detection models and their associated files"""
    models_loaded = {}
    
    # Model 1: Psychological Risk Factor Detection (Multi-class)
    try:
        model1 = load_model('final_gesture_cnn_model.h5')
        with open('final_scaler.pkl', 'rb') as f:
            scaler1 = pickle.load(f)
        with open('final_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        models_loaded['model1'] = {
            'model': model1,
            'scaler': scaler1,
            'label_encoder': label_encoder,
            'name': 'Gesture Net',
            'type': 'multi-class'
        }
        st.success("✅ Model 1 (Multi-class Gesture Detection) loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Model 1 files not found: {e}")
        models_loaded['model1'] = None
    
    # Model 2: Binary Gesture Detection
    try:
        model2 = load_model('final_binary_gesture_cnn_model.h5')
        with open('final_binary_scaler.pkl', 'rb') as f:
            scaler2 = pickle.load(f)
        
        models_loaded['model2'] = {
            'model': model2,
            'scaler': scaler2,
            'name': 'Gesture Net',
            'type': 'binary'
        }
        st.success("✅ Model 2 (Binary Gesture) loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Model 2 files not found: {e}")
        models_loaded['model2'] = None
    
    return models_loaded

def preprocess_csv(df, time_col='time'):
    """Preprocess CSV data similar to your preprocessing pipeline"""
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert time to float seconds rounded to 2 decimals (similar to your preprocessing)
    if time_col in df.columns:
        # If time is in milliseconds, convert to seconds
        if df[time_col].max() > 1000:  # Likely milliseconds
            df[time_col] = (df[time_col].astype(float) / 1000).round(2)
        else:
            df[time_col] = df[time_col].astype(float).round(2)
    
    # Drop duplicate timestamps keeping row with fewer nulls
    df = df.sort_values(by=time_col)
    df = df.loc[df.groupby(time_col).apply(lambda x: x.isnull().sum(axis=1).idxmin())]
    
    return df

def merge_hand_body_data(hands_df, body_df, time_col='time'):
    """
    Merge hands and body dataframes following the exact same logic 
    as the original preprocessing script
    """

    def custom_preprocess(df):
        """Preprocess dataframe like in original script"""
        df = df.copy()

        # Convert time to float seconds rounded to 2 decimals
        if time_col in df.columns:
            if df[time_col].max() > 1000:  # Likely milliseconds
                df[time_col] = (df[time_col].astype(float) / 1000).round(2)
            else:
                df[time_col] = df[time_col].astype(float).round(2)

        # Sort by time
        df = df.sort_values(by=time_col)

        # Drop duplicate timestamps keeping row with fewer nulls
        df = df.loc[df.groupby(time_col).apply(lambda x: x.isnull().sum(axis=1).idxmin())]

        return df

    # Step 1: Preprocess both files
    hands_processed = custom_preprocess(hands_df)
    body_processed = custom_preprocess(body_df)

    # Step 2: Merge using inner join (same as original preprocessing)
    merged_df = pd.merge(hands_processed, body_processed, on=time_col, how='inner')

    return merged_df

def extract_time_and_features(df):
    """Extract time column and feature columns from DataFrame"""
    # Identify time column (first column or column containing 'time')
    time_column = None
    if 'time' in df.columns.str.lower():
        time_column = df.columns[df.columns.str.lower().str.contains('time')][0]
    else:
        time_column = df.columns[0]  # Assume first column is time
    
    time_data = df[time_column].values
    feature_data = df.drop(columns=[time_column])
    
    return time_data, feature_data

# Model 1 Functions (Psychological Risk Factor Detection)
def create_windows(X, window_size=30, step=1):
    """Create sliding windows from the input data"""
    windows = []
    indices = []
    for i in range(0, len(X) - window_size + 1, step):
        windows.append(X[i:i+window_size])
        indices.append(i)
    return np.array(windows), indices

def predict_gestures_model1(model, X_scaled, label_encoder, window_size=30, step=1):
    """Predict gestures using Model 1 with sliding windows"""
    windows, indices = create_windows(X_scaled, window_size, step)
    probs = model.predict(windows, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    # Map predictions back to original timeline
    full_preds = np.full(len(X_scaled), -1)
    for idx, pred in zip(indices, preds):
        full_preds[idx:idx+window_size] = pred
    
    # Fill gaps in predictions
    known_idx = np.where(full_preds != -1)[0]
    for i in range(len(full_preds)):
        if full_preds[i] == -1:
            closest = known_idx[np.argmin(np.abs(known_idx - i))]
            full_preds[i] = full_preds[closest]
    
    predicted_labels = label_encoder.inverse_transform(full_preds)
    return predicted_labels

def group_gesture_intervals(df, min_duration=0.2):
    """Group consecutive gesture predictions into intervals"""
    intervals = []
    current_start = None
    current_gesture = None
    current_end = None
    
    for _, row in df.iterrows():
        t = row['time']
        g = row['predicted_gesture']
        if current_gesture is None:
            current_start = t
            current_gesture = g
            current_end = t
        elif g == current_gesture:
            current_end = t
        else:
            if current_gesture != 'unknown' and current_end - current_start >= min_duration:
                intervals.append((current_start, current_end, current_gesture))
            current_start = t
            current_gesture = g
            current_end = t
    
    if current_gesture and current_gesture != 'unknown' and current_end - current_start >= min_duration:
        intervals.append((current_start, current_end, current_gesture))
    return intervals

# Model 2 Functions (Binary Gesture Detection)
def preprocess_data_model2(feature_data, scaler, window_size=30, step=10):
    """Preprocess feature data for Model 2 (Binary Gesture Detection)"""
    # Convert to numpy array if DataFrame
    if isinstance(feature_data, pd.DataFrame):
        feature_array = feature_data.values
    else:
        feature_array = feature_data
    
    # Ensure we have the correct number of features (258 for Model 2 - after time column removal)
    if feature_array.shape[1] != 258:
        if feature_array.shape[1] > 258:
            feature_array = feature_array[:, :258]
        else:
            padding = np.zeros((feature_array.shape[0], 258 - feature_array.shape[1]))
            feature_array = np.hstack([feature_array, padding])
    
    # Convert to float32 and handle non-numeric data
    try:
        feature_array = feature_array.astype(np.float32)
    except ValueError:
        feature_df = pd.DataFrame(feature_array)
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        feature_array = feature_df.fillna(0).values.astype(np.float32)
    
    # Handle NaN and infinite values
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create sliding windows
    windows = []
    
    for i in range(0, len(feature_array) - window_size + 1, step):
        window = feature_array[i:i+window_size]
        windows.append(window)
    
    if len(windows) == 0:
        return None
    
    windows = np.array(windows)
    
    # Normalize the data using the trained scaler
    n_samples, n_frames, n_features = windows.shape
    windows_flat = windows.reshape(-1, n_features)
    windows_scaled = scaler.transform(windows_flat).reshape(n_samples, n_frames, n_features)
    
    return windows_scaled

def predict_gestures_model2(model, X_data):
    """Predict gestures using the binary model"""
    predictions = model.predict(X_data, verbose=0)
    gesture_predictions = (predictions > 0.5).astype(int).flatten()
    confidence_scores = predictions.flatten()
    return gesture_predictions, confidence_scores

def map_predictions_to_timeline(time_data, predictions, window_size=30, step=10):
    """Map windowed predictions back to original timeline"""
    timeline_predictions = np.zeros(len(time_data))
    timeline_times = []
    
    for i, pred in enumerate(predictions):
        start_idx = i * step
        middle_idx = start_idx + window_size // 2
        if middle_idx < len(time_data):
            timeline_times.append(time_data[middle_idx])
            timeline_predictions[middle_idx] = pred
    
    return np.array(timeline_times), timeline_predictions[:len(timeline_times)]

def find_gesture_ranges(times, predictions, min_gap=0.2):
    """Find continuous gesture ranges from predictions"""
    gesture_ranges = []
    
    if len(predictions) == 0:
        return gesture_ranges
    
    gesture_indices = np.where(predictions == 1)[0]
    
    if len(gesture_indices) == 0:
        return gesture_ranges
    
    start_idx = gesture_indices[0]
    prev_idx = gesture_indices[0]
    
    for i in range(1, len(gesture_indices)):
        current_idx = gesture_indices[i]
        time_gap = times[current_idx] - times[prev_idx]
        
        if time_gap > min_gap:
            end_idx = prev_idx
            gesture_ranges.append((times[start_idx], times[end_idx]))
            start_idx = current_idx
        
        prev_idx = current_idx
    
    gesture_ranges.append((times[start_idx], times[prev_idx]))
    return gesture_ranges

def create_gesture_timeline_plot(times, predictions, confidence_scores, title="Gesture Detection Timeline"):
    """Create an interactive timeline plot showing gesture detections"""
    plot_df = pd.DataFrame({
        'Time': times,
        'Gesture': predictions,
        'Confidence': confidence_scores
    })
    
    fig = go.Figure()
    
    # Add confidence scores as a line
    fig.add_trace(go.Scatter(
        x=plot_df['Time'],
        y=plot_df['Confidence'],
        mode='lines',
        name='Confidence Score',
        line=dict(color='#2E86C1', width=2),
        yaxis='y2'
    ))
    
    # Add gesture detections as markers
    gesture_times = plot_df[plot_df['Gesture'] == 1]['Time']
    gesture_confidences = plot_df[plot_df['Gesture'] == 1]['Confidence']
    
    if len(gesture_times) > 0:
        fig.add_trace(go.Scatter(
            x=gesture_times,
            y=[1] * len(gesture_times),
            mode='markers',
            name='Detected Gestures',
            marker=dict(
                color='#E74C3C',
                size=12,
                symbol='diamond',
                line=dict(color='#C0392B', width=2)
            ),
            hovertemplate='<b>Gesture Detected</b><br>Time: %{x:.2f}s<br>Confidence: %{customdata:.3f}<extra></extra>',
            customdata=gesture_confidences
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Gesture Detection',
        yaxis2=dict(
            title='Confidence Score',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=['No Gesture', 'Gesture'],
            range=[-0.5, 1.5]
        ),
        height=400,
        hovermode='x unified'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠👋 Gesture Net</h1>', unsafe_allow_html=True)
    st.markdown("### Upload hands and body CSV files to get predictions from both models")
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload CSV Files**: Upload both hands and body CSV files
    2. **Auto-Merge**: Files will be automatically merged on time column
    3. **Dual Processing**: The merged data will be processed by both models
    4. **View Results**: See results from both models:
       - **Model 1**: Multi-class Gesture Detection
       - **Model 2**: Binary Gesture Detection
    5. **Compare Outputs**: Analyze differences between model predictions
    """)
    
    st.sidebar.header("⚙️ Model 2 Settings")
    confidence_threshold = st.sidebar.slider("Binary Model Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    min_gap = st.sidebar.slider("Minimum Gap Between Gestures (seconds)", 0.1, 2.0, 0.2, 0.1)
    
    st.sidebar.header("⚙️ Model 1 Settings")
    min_duration = st.sidebar.slider("Minimum Gesture Duration (seconds)", 0.1, 1.0, 0.2, 0.1)
    
    # Load models
    with st.spinner("Loading trained models..."):
        models = load_trained_models()
    
    if models['model1'] is None and models['model2'] is None:
        st.error("Could not load any models. Please check if the model files are available.")
        return
    
    # File upload section
    st.header("📁 Upload CSV Files")
    st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👋 Hands Data")
        hands_file = st.file_uploader(
            "Choose hands CSV file",
            type=['csv'],
            key="hands",
            help="Upload the CSV file containing hand gesture data"
        )
    
    with col2:
        st.subheader("🏃 Body Data")
        body_file = st.file_uploader(
            "Choose body CSV file",
            type=['csv'],
            key="body",
            help="Upload the CSV file containing body gesture data"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process files when both are uploaded
    if hands_file is not None and body_file is not None:
        try:
            # Read the CSV files
            hands_df = pd.read_csv(hands_file)
            body_df = pd.read_csv(body_file)
            
            st.markdown('<div class="merge-info">', unsafe_allow_html=True)
            st.info("📊 **Files uploaded successfully!** Merging hands and body data...")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display info about uploaded files
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Hands Data Info:**")
                st.write(f"- Rows: {len(hands_df)}")
                st.write(f"- Columns: {len(hands_df.columns)}")
            
            with col2:
                st.write("**Body Data Info:**")
                st.write(f"- Rows: {len(body_df)}")
                st.write(f"- Columns: {len(body_df.columns)}")
            
            # Merge the data
            with st.spinner("Merging hands and body data..."):
                merged_df = merge_hand_body_data(hands_df, body_df)
            
            st.success(f"✅ **Data merged successfully!** Combined dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns.")
            
            # Extract time and features from merged data
            time_data, feature_data = extract_time_and_features(merged_df)
            
            # Display basic info about the merged data
            st.header("📊 Merged Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Rows", len(merged_df))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Feature Columns", len(feature_data.columns))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                duration = time_data[-1] - time_data[0] if len(time_data) > 1 else 0
                st.metric("Duration", f"{duration:.2f}s")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                overlap_percentage = (len(merged_df) / min(len(hands_df), len(body_df))) * 100
                st.metric("Data Overlap", f"{overlap_percentage:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show merged data preview
            st.subheader("Merged Data Preview")
            st.dataframe(merged_df.head(), use_container_width=True)
            
            # Download merged data option
            csv_buffer = io.StringIO()
            merged_df.to_csv(csv_buffer, index=False)
            merged_csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="💾 Download Merged Dataset",
                data=merged_csv_data,
                file_name=f"merged_hands_body_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Process with both models
            st.header("🔄 Processing with Both Models")
            
            # Model 1 Processing
            if models['model1'] is not None:
                st.markdown('<div class="model-section model-1-section">', unsafe_allow_html=True)
                st.subheader("🧠 Model 1: Multi-class Gesture Detection")
                
                with st.spinner("Processing with Model 1..."):
                    try:
                        # Scale feature data for Model 1
                        scaled_data = models['model1']['scaler'].transform(feature_data.values)
                        
                        # Predict gestures
                        predicted_labels = predict_gestures_model1(
                            models['model1']['model'], 
                            scaled_data, 
                            models['model1']['label_encoder']
                        )
                        
                        # Create results DataFrame
                        result_df = pd.DataFrame({
                            'time': time_data[:len(predicted_labels)],
                            'predicted_gesture': predicted_labels
                        })
                        result_df = result_df[result_df['predicted_gesture'] != 'unknown']
                        
                        # Group gesture intervals
                        intervals = group_gesture_intervals(result_df, min_duration)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Unique Gestures", len(set(predicted_labels)))
                        with col2:
                            st.metric("Gesture Intervals", len(intervals))
                        
                        st.subheader("⏱️ Detected Gesture Time Ranges")
                        if intervals:
                            for start, end, gesture in intervals:
                                st.markdown(
                                    f'<div class="gesture-range">{start:.2f}s - {end:.2f}s → <strong>{gesture}</strong></div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Download button for Model 1
                            range_data = pd.DataFrame(intervals, columns=['Start Time', 'End Time', 'Gesture'])
                            csv_range = range_data.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download Model 1 Results",
                                data=csv_range,
                                file_name=f'model1_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                                mime='text/csv'
                            )
                        else:
                            st.info("No gesture intervals detected by Model 1.")
                        
                    except Exception as e:
                        st.error(f"Error processing with Model 1: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Model 2 Processing
            if models['model2'] is not None:
                st.markdown('<div class="model-section model-2-section">', unsafe_allow_html=True)
                st.subheader("👋 Model 2: Binary Gesture Detection")
                
                with st.spinner("Processing with Model 2..."):
                    try:
                        # Preprocess data for Model 2
                        X_processed = preprocess_data_model2(feature_data, models['model2']['scaler'])
                        
                        if X_processed is not None:
                            # Make predictions
                            predictions, confidence_scores = predict_gestures_model2(models['model2']['model'], X_processed)
                            
                            # Map predictions back to timeline
                            timeline_times, timeline_predictions = map_predictions_to_timeline(time_data, predictions)
                            
                            # Apply confidence threshold
                            thresholded_predictions = (confidence_scores > confidence_threshold).astype(int)
                            
                            # Find gesture ranges
                            gesture_ranges = find_gesture_ranges(timeline_times, thresholded_predictions, min_gap)
                            
                            # Display results
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Gestures", len(gesture_ranges))
                            
                            with col2:
                                total_gesture_time = sum(end - start for start, end in gesture_ranges)
                                st.metric("Total Gesture Time", f"{total_gesture_time:.2f}s")
                            
                            with col3:
                                avg_confidence = np.mean(confidence_scores[thresholded_predictions == 1]) if np.any(thresholded_predictions) else 0
                                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                            
                            with col4:
                                gesture_percentage = (np.sum(thresholded_predictions) / len(thresholded_predictions)) * 100
                                st.metric("Gesture %", f"{gesture_percentage:.1f}%")
                            
                            # Display gesture time ranges
                            st.subheader("⏰ Binary Gesture Time Ranges")
                            if gesture_ranges:
                                for i, (start_time, end_time) in enumerate(gesture_ranges, 1):
                                    if abs(start_time - end_time) < 0.1:
                                        time_range = f"{start_time:.2f}s"
                                    else:
                                        time_range = f"{start_time:.2f}s - {end_time:.2f}s"
                                    
                                    st.markdown(
                                        f'<div class="gesture-range"><strong>Gesture {i}:</strong> {time_range}</div>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Create downloadable results for Model 2
                                results_df = pd.DataFrame({
                                    'Gesture_Number': range(1, len(gesture_ranges) + 1),
                                    'Start_Time': [start for start, end in gesture_ranges],
                                    'End_Time': [end for start, end in gesture_ranges],
                                    'Duration': [end - start for start, end in gesture_ranges]
                                })
                                
                                csv_buffer = io.StringIO()
                                results_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="📥 Download Model 2 Results",
                                    data=csv_data,
                                    file_name=f"model2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # Interactive timeline plot
                                st.subheader("📈 Binary Gesture Detection Timeline")
                                timeline_fig = create_gesture_timeline_plot(
                                    timeline_times, thresholded_predictions, confidence_scores,
                                    "Model 2: Binary Gesture Detection Timeline"
                                )
                                st.plotly_chart(timeline_fig, use_container_width=True)
                                
                            else:
                                st.info("No gestures detected by Model 2 with the current confidence threshold.")
                        else:
                            st.error("Could not process the data for Model 2.")
                    
                    except Exception as e:
                        st.error(f"Error processing with Model 2: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Comparison Section
            st.header("🔄 Model Comparison")
            st.info("This section compares the outputs from both models to help you understand the differences in their predictions.")
            
            # If both models processed successfully, show comparison
            if models['model1'] is not None and models['model2'] is not None:
                st.subheader("Key Differences")
                st.markdown("""
                - **Model 1** performs multi-class classification and identifies specific gesture types
                - **Model 2** performs binary classification (gesture vs no gesture)
                - **Model 1** uses the raw feature count from your merged data
                - **Model 2** expects exactly 258 features (after time column removal)
                - Both models process the same merged hands+body data with the time column removed
                """)
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.error("Please make sure your CSV files have the correct format with time columns that can be merged.")
    
    elif hands_file is not None or body_file is not None:
        st.warning("⚠️ Please upload both hands and body CSV files to proceed with the analysis.")
        if hands_file is not None:
            st.info("✅ Hands file uploaded successfully. Waiting for body file...")
        if body_file is not None:
            st.info("✅ Body file uploaded successfully. Waiting for hands file...")

if __name__ == "__main__":
    main()