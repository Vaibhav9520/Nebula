import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = r"C:\Users\Lenovo\Desktop\NEBULA\Dataset.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns
df_numeric = df.drop(columns=["Country", "Energy Type"])

# Define features (X) and target variable (y)
X = df_numeric.drop(columns=["Energy Consumption"])
y = df_numeric["Energy Consumption"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Get feature importances (ADD THIS)
feature_importances = model.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(feature_importances)[::-1]  # Sort by importance

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Save plots as images
plt.figure(figsize=(12, 8))  # Increase figure size
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap", fontsize=16)  # Larger title
plt.xticks(rotation=45)  # Rotate labels to avoid overlap
plt.yticks(rotation=0)
plt.tight_layout()  # Auto-adjust layout to prevent cutting
plt.savefig("heatmap.png", dpi=300)  # Increase resolution
plt.close()


plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")
plt.savefig("actual_vs_predicted.png")  # Save the image
plt.close()

plt.figure(figsize=(12, 8))  # Increase figure size
sns.barplot(x=feature_importances[sorted_indices], y=np.array(feature_names)[sorted_indices], palette="viridis")
plt.xlabel("Feature Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.title("Feature Importance in Prediction Model", fontsize=18, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()  # Adjust layout to prevent text cutting
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')  # High resolution & crop extra space
plt.close()

plt.figure(figsize=(10,6))
sns.histplot(y, bins=30, kde=True, color="Blue")  # KDE for smooth curve
plt.xlabel("Energy Consumption")
plt.ylabel("Frequency")
plt.title("Distribution of Energy Consumption")
plt.savefig("energy_consumption_distribution.png")
plt.close()