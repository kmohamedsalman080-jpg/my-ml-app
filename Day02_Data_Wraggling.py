import numpy as np
import pandas as pd

# 1. Dynamic Log/Data Generator using NumPy Vectorization
def generate_raw_sensor_data(num_records: int = 10) -> pd.DataFrame:
    np.random.seed(42)  # For reproducible results
    
    # Generating mock AI Sensor Data
    timestamps = pd.date_range(start="2026-10-02 10:00", periods=num_records, freq="min")
    sensor_ids = np.random.choice(["SENSOR_01", "SENSOR_02", "SENSOR_03"], size=num_records)
    temperatures = np.random.normal(loc=45.0, scale=12.0, size=num_records)  # Celsius
    cpu_loads = np.random.uniform(low=10.0, high=100.0, size=num_records)   # Percentage
    
    # Intentionally introducing missing values (NaN) to test cleaning
    temperatures[np.random.choice(num_records, size=2, replace=False)] = np.nan

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Sensor_ID": sensor_ids,
        "Temperature_C": temperatures,
        "CPU_Load": cpu_loads
    })
    return df

# 2. Data Cleaning & Wrangling Pipeline
def wrangle_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 🔍 RAW DATA ---")
    print(df, "\n")

    # Handling Missing Values: Fill missing temperatures with the median value
    median_temp = df["Temperature_C"].median()
    df["Temperature_C"] = df["Temperature_C"].fillna(median_temp)

    # Vectorized Computation: Convert Temperature Celsius to Fahrenheit using NumPy logic
    df["Temperature_F"] = (df["Temperature_C"] * 9/5) + 32

    # Filtering Data: High load alert (CPU_Load > 70%)
    high_load_mask = df["CPU_Load"] > 70.0
    df["High_Load_Alert"] = high_load_mask

    print("--- ✅ CLEANED & PROCESSED DATA ---")
    print(df, "\n")

    # Aggregation: Grouping by Sensor_ID
    summary = df.groupby("Sensor_ID").agg(
        Avg_Temp_F=("Temperature_F", "mean"),
        Max_CPU_Load=("CPU_Load", "max"),
        Alert_Count=("High_Load_Alert", "sum")
    ).reset_index()

    print("--- 📊 SENSOR SUMMARY REPORT ---")
    print(summary)
    return df

if __name__ == "__main__":
    raw_df = generate_raw_sensor_data(12)
    cleaned_df = wrangle_sensor_data(raw_df)