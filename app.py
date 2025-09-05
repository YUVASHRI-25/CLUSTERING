import gradio as gr
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset (Wine CSV with 13 features)
data = pd.read_csv("E:/PYTHON/UNSUPERVISED/new_updated_wine-clustering-1000.csv")  # change path if needed

# Features used
features = [
    "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity", "Magnesium",
    "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols",
    "Proanthocyanins", "Color_Intensity", "Hue", "OD280", "Proline"
]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Train KMeans model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled_data)

# Store shifted cluster labels (1,2,3 instead of 0,1,2)
data["Cluster"] = kmeans.labels_ + 1  

# Prediction function
def predict_cluster(Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium,
                    Total_Phenols, Flavanoids, Nonflavanoid_Phenols,
                    Proanthocyanins, Color_Intensity, Hue, OD280, Proline):
    
    input_df = pd.DataFrame([[Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium,
                              Total_Phenols, Flavanoids, Nonflavanoid_Phenols,
                              Proanthocyanins, Color_Intensity, Hue, OD280, Proline]],
                            columns=features)

    # Scale input
    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]

    # Shift from 0–2 → 1–3
    return f"Predicted Cluster: {cluster + 1}"

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center'>🍷 Wine Clustering App</h1>")
    gr.Markdown("Input wine features to predict the cluster using **KMeans**")

    with gr.Row():
        with gr.Column():
            Alcohol = gr.Number(label="Alcohol", value=13.86)
            Malic_Acid = gr.Number(label="Malic_Acid", value=1.35)
            Ash = gr.Number(label="Ash", value=2.27)
            Ash_Alcanity = gr.Number(label="Ash_Alcanity", value=16)
            Magnesium = gr.Number(label="Magnesium", value=98)
            Total_Phenols = gr.Number(label="Total_Phenols", value=2.98)
            Flavanoids = gr.Number(label="Flavanoids", value=3.15)
            Nonflavanoid_Phenols = gr.Number(label="Nonflavanoid_Phenols", value=0.22)
            Proanthocyanins = gr.Number(label="Proanthocyanins", value=1.85)
            Color_Intensity = gr.Number(label="Color_Intensity", value=7.22)
            Hue = gr.Number(label="Hue", value=1.01)
            OD280 = gr.Number(label="OD280", value=3.55)
            Proline = gr.Number(label="Proline", value=1045)

            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            output = gr.Textbox(label="Output")

    # Connect buttons
    submit_btn.click(
        predict_cluster,
        inputs=[Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium,
                Total_Phenols, Flavanoids, Nonflavanoid_Phenols,
                Proanthocyanins, Color_Intensity, Hue, OD280, Proline],
        outputs=output
    )

    clear_btn.click(lambda: "", None, output)

# Launch app
demo.launch()
