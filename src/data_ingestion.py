import pandas as pd
import os

def load_data(file_path):
    try:
        # Data load karna
        df = pd.read_csv(file_path)
        print("✅ Data Load Ho Gaya!")
        
        # Data ki basic info print karna
        print(f"Dataset Shape: {df.shape}")
        print("\nColumn Names:")
        print(df.columns.tolist())
        
        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Path set karein (data folder ke andar)
    path = os.path.join("data", "ai4i2020.csv")
    data = load_data(path)
    
    if data is not None:
        print("\nPehli 5 rows:")
        print(data.head())