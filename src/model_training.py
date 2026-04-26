import pandas as pd
from sklearn.ensemble import RandomForestClassifier #Humne us "Mistry" (Algorithm) ko bulaya jisne deewar (logic) banani hai.
from sklearn.metrics import accuracy_score,f1_score #Yeh woh "Teachers" hain jo check karenge ke model ne imtehaan mein kitne marks liye.
import mlflow #Yeh sabse zaroori cheez hai. MLflow ek "Digital Diary" hai jo record karti hai ke tumne kaunsa model banaya, uski accuracy kya thi, aur uski settings kya thin.
import mlflow
import os
from data_preprocessing import preprocess_data # Humne pichli file se kaha: "Bhai, apna woh safai wala function mujhe udhaar de do."
def train_model():
    #Agar tumne apne code mein rasta likh diya "data/ai4i2020.csv" (Linux style), aur tumhara koi dost ya koi server usse Windows par chalayega, toh code Crash ho jayega kyunki Windows ko woh slash samajh nahi aayega.
    #"Bhai, ek folder hai 'data' aur uske andar ek file hai 'ai4i2020.csv'. Tum check karo ke main kaunsa computer use kar raha hoon, aur mere computer ke hisaab se sahi wala slash ( / ya \ ) laga kar mujhe poora rasta bana do."
    path=os.path.join("data","ai4i2020.csv")  
    #os: Iska matlab hai "Operating System". Yeh Python ki woh library hai jo computer ke system se baatein karti hai.
    #.path: Yeh os ke andar ek module hai jo sirf rasta (address) handle karne ke liye bana hai.
    #.join: Iska matlab hai "Jodna" (Combine karna).
    X_train, X_test, y_train, y_test = preprocess_data(path)
    mlflow.set_experiment("Predictive_Maintenance_Project") #Hum Diary mein ek naya chapter shuru kar rahe hain jiska naam hai "Predictive_Maintenance_Project".
    with mlflow.start_run():#Iska matlab hai: "Recording Shuru!" Ab iske niche jo bhi hoga, MLflow usse note karta jayega.
        n_estimators = 200  # Kitne Trees (Experts) honge?
        max_depth = 15    # Har Tree kitna gehra (Detailed) sawal puchega?
        #"Bhai, jitne zyada sawal utni achi prediction, toh depth 100 kyun nahi rakh dete?"
        #Yahan ek masla aata hai jise ML mein kehte hain Overfitting (Ratta Maarna).
        rf=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=42,class_weight='balanced' ) 
        rf.fit(X_train,y_train) #Ab humne apne "Mistry" (Model) ko training data diya hai, jisse wo seekh sake.
        y_pred=rf.predict(X_test) #Humne model ko woh 20 sawal (X_test) diye jo usne pehle nahi dekhe the. Model ne apne "Andaze" (Predictions) diye.
        acc=accuracy_score(y_test,y_pred) #Ab humne apne "Teachers" (Metrics) se poocha: "Bhai, tumhare hisaab se model ne kitne marks liye?"
        f1 = f1_score(y_test, y_pred)
        print(f"✅ Model Trained! Accuracy: {acc:.2f}, F1-Score: {f1:.2f}")
        mlflow.log_param("n_estimators", n_estimators) #Ab humne MLflow ko bataya ke humne kitne Trees (Experts) rakhe the.
        mlflow.log_param("max_depth", max_depth) #Ab humne MLflow ko bataya ke har Tree kitna gehra (Detailed) sawal puchega.
        mlflow.log_metric("accuracy", acc) #Ab humne MLflow ko bataya ke model ki accuracy kya thi.
        mlflow.log_metric("f1_score", f1) #Ab humne MLflow ko bataya ke model ki f1-score kya thi.
        mlflow.sklearn.log_model(rf, "random_forest_model") #Poora ka poora Model (Trained Brain) hi save kar lo taake baad mein use kar sakein.
        print("📝 Sab kuch MLflow mein record ho gaya hai!")
if __name__ == "__main__":
    train_model()