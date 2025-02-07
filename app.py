from flask import Flask, send_file
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

app = Flask(__name__)

# Load dataset
file_path = "Dataset.csv"
df = pd.read_csv(file_path)

@app.route('/plot/heatmap')
def heatmap():
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("heatmap.png")
    return send_file("heatmap.png", mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
