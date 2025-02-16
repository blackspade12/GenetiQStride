from fastapi import FastAPI, HTTPException
import numpy as np
import tensorflow as tf
import joblib
from pydantic import BaseModel
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
import os


# Initialize FastAPI app
app = FastAPI()

# Load trained models
race_potential_model = joblib.load("race_model.pkl")
injury_risk_model = joblib.load("injury_model.pkl")
breeding_score_model = tf.keras.models.load_model("breeding_cnn_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
# Load PCA model
pca = joblib.load("pca_model.pkl")
X_train_scaled = joblib.load("X_train_scaled.pkl")  # Load saved scaled data


# Define request model
class HorseFeatures(BaseModel):
    Breed: str
    Sex: str
    MSTN_Gene: str
    PPARδ_Gene: str
    COL1A1_Risk: str
    ACTN3_Type: str
    Distance_Pref: str
    Champion_Lineage: str
    Injury_History: str
    Speed_Index: float
    Stamina_Index: float

# Preprocessing function
def preprocess_input(data: HorseFeatures):
    try:
        feature_vector = []

        # Encode categorical values safely
        for col in ["Breed", "Sex", "MSTN_Gene", "PPARδ_Gene", "COL1A1_Risk", "ACTN3_Type", "Distance_Pref"]:
            if data.dict()[col] in label_encoders[col].classes_:
                encoded_value = label_encoders[col].transform([data.dict()[col]])[0]
            else:
                encoded_value = 0  # Assign default value for unseen labels
            feature_vector.append(encoded_value)

        # Convert Yes/No to binary
        feature_vector.append(1 if data.Champion_Lineage.lower() == "yes" else 0)
        feature_vector.append(1 if data.Injury_History.lower() == "yes" else 0)

        # Add numerical features
        feature_vector.append(data.Speed_Index)
        feature_vector.append(data.Stamina_Index)

        # **Ensure correct feature count (Padding for missing features)**
        while len(feature_vector) < 13:
            feature_vector.append(0)  # Adds default values for missing features

        # Convert to NumPy array & scale
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector = scaler.transform(feature_vector)  # Scale input
        return feature_vector

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")


@app.get("/")
def home():
    return {"message": "✅ GenetiQStride API is running!"}

@app.post("/predict_race_potential")
def predict_race_potential(features: HorseFeatures):
    input_data = preprocess_input(features)
    prediction = race_potential_model.predict(input_data)[0]
    return {"Race_Potential": round(float(prediction), 2)}

@app.post("/predict_injury_risk")
def predict_injury_risk(features: HorseFeatures):
    input_data = preprocess_input(features)
    prediction = injury_risk_model.predict(input_data)[0]
    return {"Injury_Risk": round(float(prediction), 2)}

@app.post("/predict_breeding_score")
def predict_breeding_score(features: HorseFeatures):
    input_data = preprocess_input(features)
    prediction = breeding_score_model.predict(np.expand_dims(input_data, axis=-1))[0][0]
    prediction = prediction * 100  # ✅ Convert back to 0-100 range
    prediction = max(0, min(100, prediction))  # ✅ Ensure it's within 0-100
    return {"Breeding_Score": round(float(prediction), 2)}


@app.get("/visualize_pca")
def visualize_pca():
    try:
        # Apply PCA transformation
        X_pca = pca.transform(X_train_scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='blue', edgecolors='k')
        plt.xlabel("PC1(Champion Genetic Index)")
        plt.ylabel("PC2(Performance & Injury Risk Index)")
        plt.title("Genetic Data PCA Visualization")
        plt.grid(True)

        # Save the plot
        plot_path = "pca_plot.png"
        plt.savefig(plot_path)
        plt.close()

        return FileResponse(plot_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PCA plot: {str(e)}")

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
