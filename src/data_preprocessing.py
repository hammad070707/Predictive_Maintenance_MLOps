# Ab hum src/data_preprocessing.py file banayenge. Isme hum 3 kaam karenge:
# Faltu Columns Hatana: UDI aur Product ID sirf counting ke liye hain, inka machine kharab hone se koi lena dena nahi. Unhe nikal denge.
# Encoding: Machine ko "L", "M", "H" (Quality types) samajh nahi aati, unhe numbers (0, 1, 2) mein badlenge.
#Split: Data ko "Training" (seekhne ke liye) aur "Testing" (test karne ke liye) mein banten ge.
import pandas as pd
from sklearn.model_selection import train_test_split #Data ke do hisse karne ke liye (Ek seekhne ke liye, ek imtehaan lene ke liye).

from sklearn.preprocessing import LabelEncoder #English words ko numbers mein badalne ke liye
import os

def preprocess_data(input_path):
    # 1. Data Load karein
    df = pd.read_csv(input_path)
    
    # 2. Faltu columns drop karein (UDI, Product ID)
    # Inki zaroorat prediction mein nahi hoti
    df = df.drop(['UDI', 'Product ID',"TWF", "HDF", "PWF",'OSF', 'RNF' ], axis=1)
    
    # 3. Label Encoding (Type column: L, M, H ko numbers mein badlo)
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    
    # 4. Features (X) aur Target (y) alag karein
    # 'Machine failure' hamara target hai jo humne predict karna hai
    X = df.drop('Machine failure', axis=1)
    y = df['Machine failure']
    
    # 5. Train-Test Split (80% seekhne ke liye, 20% test karne ke liye)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("✅ Data Preprocessing Mukammal!")
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    path = os.path.join("data", "ai4i2020.csv")
    X_train, X_test, y_train, y_test = preprocess_data(path)

