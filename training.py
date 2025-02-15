import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
import joblib

# Load dataset
df = pd.read_csv("GenetiQStride_Dataset_30000.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Breed", "Sex", "MSTN_Gene", "PPARδ_Gene", "COL1A1_Risk", "ACTN3_Type", "Distance_Pref", "Champion_Lineage", "Injury_History"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Select features and target variables
X = df.drop(columns=["Horse_ID", "Sire_ID", "Dam_ID", "Race_Potential", "Injury_Risk", "Breeding_Score"])
y_race = df["Race_Potential"]
y_injury = df["Injury_Risk"]
y_breeding = df["Breeding_Score"]

# Train-test split
X_train, X_test, y_race_train, y_race_test = train_test_split(X, y_race, test_size=0.2, random_state=42)
X_train, X_test, y_injury_train, y_injury_test = train_test_split(X, y_injury, test_size=0.2, random_state=42)
X_train, X_test, y_breeding_train, y_breeding_test = train_test_split(X, y_breeding, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# PCA for genetic trend visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
joblib.dump(pca, "pca_model.pkl")

# Train Random Forest for Race Potential
race_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
race_model.fit(X_train_scaled, y_race_train)
race_accuracy = race_model.score(X_test_scaled, y_race_test)
print(f"Race Potential Model Accuracy: {race_accuracy:.2f}")

# Train Random Forest for Injury Risk
injury_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
injury_model.fit(X_train_scaled, y_injury_train)
injury_accuracy = injury_model.score(X_test_scaled, y_injury_test)
print(f"Injury Risk Model Accuracy: {injury_accuracy:.2f}")

# CNN for SNP-based Breeding Recommendation
X_breeding_train = np.expand_dims(X_train_scaled, axis=-1)
X_breeding_test = np.expand_dims(X_test_scaled, axis=-1)

cnn_model = keras.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_breeding_train.shape[1], 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

cnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
cnn_model.fit(X_breeding_train, y_breeding_train, epochs=20, batch_size=64, validation_data=(X_breeding_test, y_breeding_test))
cnn_accuracy = cnn_model.evaluate(X_breeding_test, y_breeding_test, verbose=0)[1]
print(f"Breeding Recommendation Model MAE: {cnn_accuracy:.2f}")

# Save models
joblib.dump(race_model, "race_model.pkl")
joblib.dump(injury_model, "injury_model.pkl")
cnn_model.save("breeding_cnn_model.keras")  # ✅ Uses recommended Keras format

print("✅ Models trained and saved successfully!")