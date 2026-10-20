import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style setup for clean visuals
sns.set_theme(style="darkgrid")

def generate_sensor_dataset(num_samples: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    
    # Simulating 200 log entries
    cpu_temp = np.random.normal(loc=65, scale=10, size=num_samples)
    
    # Correlation: High Temp usually means Higher CPU Load + Noise
    cpu_load = (cpu_temp * 1.2) + np.random.normal(loc=0, scale=15, size=num_samples)
    cpu_load = np.clip(cpu_load, 10, 100) # Keep within 10%-100%
    
    # Memory Usage in MB
    ram_usage = np.random.uniform(low=2000, high=8000, size=num_samples)
    
    # Injecting a few anomalous outlier points
    cpu_temp[::40] += 35 

    df = pd.DataFrame({
        "CPU_Temp_C": cpu_temp,
        "CPU_Load_Pct": cpu_load,
        "RAM_Usage_MB": ram_usage
    })
    return df

def perform_eda(df: pd.DataFrame):
    print("--- 📋 STATISTICAL SUMMARY ---")
    print(df.describe().T, "\n")

    # Creating a 1x3 Subplot Grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Day 3: AI Hardware Metrics EDA & Anomaly Detection", fontsize=16)

    # 1. Distribution Plot (Histogram + KDE)
    sns.histplot(df["CPU_Temp_C"], kde=True, ax=axes[0], color="crimson")
    axes[0].set_title("CPU Temperature Distribution")
    axes[0].set_xlabel("Temperature (°C)")

    # 2. Correlation Scatter Plot
    sns.scatterplot(x="CPU_Temp_C", y="CPU_Load_Pct", data=df, ax=axes[1], color="teal", alpha=0.7)
    sns.regplot(x="CPU_Temp_C", y="CPU_Load_Pct", data=df, ax=axes[1], scatter=False, color="darkorange")
    axes[1].set_title("CPU Load vs Temperature Correlation")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("CPU Load (%)")

    # 3. Correlation Heatmap
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[2])
    axes[2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = generate_sensor_dataset()
    perform_eda(data)

