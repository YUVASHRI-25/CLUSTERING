import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("new_updated_wine-clustering-1000.csv")

# Features
features = [
    "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity", "Magnesium",
    "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280", "Proline"
]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=50)
kmeans.fit(scaled_data)
data["Cluster"] = kmeans.labels_ + 1  

# Prediction function
def predict_cluster(*inputs):
    input_df = pd.DataFrame([inputs], columns=features)
    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]
    return f"Predicted Cluster: {cluster + 1}"

# Build Gradio app
with gr.Blocks() as demo:

    # Header
    gr.HTML("""
    <div style="text-align:center; margin-bottom:25px;">
        <h1 style="color:#8B0000; font-family: Arial, sans-serif;">🍷 Wine Clustering App</h1>
        <p style="font-size:16px; color:#333;">Enter wine features to predict its cluster using <b>KMeans</b></p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            input_fields = []
            default_values = [13.86,1.35,2.27,16,98,2.98,3.15,0.22,1.85,7.22,1.01,3.55,1045]
            for feat, val in zip(features, default_values):
                input_fields.append(gr.Number(label=feat, value=val))
            
            submit_btn = gr.Button("Predict Cluster")
            clear_btn = gr.Button("Clear Inputs")

        with gr.Column():
            output = gr.Textbox(label="Prediction Output", interactive=False)

    # Button actions
    submit_btn.click(predict_cluster, inputs=input_fields, outputs=output)
    clear_btn.click(lambda: [0]*len(input_fields), None, input_fields)

    # Footer
    gr.HTML("""
    <div style="text-align:center; margin-top:30px; font-size:14px; color:gray;">
        Developed with ❤️ using Gradio & scikit-learn
    </div>
    """)

demo.launch()

