# MLOps mein hum yehi kar rahe hain:
# Hum computer se keh rahe hain ke: "Bhai, MLflow ki diary kholo, purane best model ki accuracy dekho aur naye wale ki dekho.
#  Agar naya wala behtar hai, "
# "toh uspar 'BEST' ki mohar (stamp) laga do."
# HUM IS FILE MAI YEH 3 KAAM KRAIN GY
# 1. Diary Khulwana (MLflow Client),Hum MLflow se kehte hain: "Bhai, mujhe saare models ka record dikhao jo ab tak train hue hain."
# Scores ka Muqabla (Comparison)
# Mohar Lagana (Model Registry),Jeetne wale model ko hum ek khaas jagah rakh dete hain jise "Model Registry" kehte hain. Ye ek "VIP Room" witnessed hai jahan sirf sabse ache models rakhe jate hain.

import mlflow
from mlflow.tracking import MlflowClient #MlflowClient ek manager ki tarah hai. Iska kaam hai diary ke andar ja kar purani entries ko dhoondna aur unhe nikalna.
import shutil # Files aur folders ko copy/move karne ke liye
import os # Folder rasta (path) check karne ke liye

def faisla_sunao():
    client=MlflowClient() #s line ka matlab hai ke humne MLflow ke database se raabta (connect) karne ke liye ek Manager (Object) tayyar kar liya hai. Ab hum is client ke zariye diary se sawal puch sakte hain.
    
    # --- EXPERIMENT ID DHOONDNA ---
    # Humne naam se ID nikali taake FileNotFoundError na aaye
    exp_name = "Predictive_Maintenance_Project"
    experiment = mlflow.get_experiment_by_name(exp_name)
    exp_id = experiment.experiment_id

    run=client.search_runs(
       experiment_ids=[exp_id], #Hum client se keh rahe hain: "Bhai, is chapter ke andar ke sabhi entries (runs) dikhao."
       order_by=["metrics.accuracy DESC"]#Ab hum client se keh rahe hain: "Bhai, sabhi entries ko unki accuracy ke hisaab se descending order mein arrange kar do, taake sabse achi entry sabse pehle aaye."
    )
    
    if len(run) > 0:
        winner_run = run[0]
        run_id = winner_run.info.run_id
        
        # --- MUQABLA LOGIC ---
        if len(run) > 1:
            naya_score = run[0].data.metrics["accuracy"] #Yeh line ka matlab hai ke humne sabse pehle entry (jo sabse achi hai) se uski accuracy nikal li.
            purana_score = run[1].data.metrics["accuracy"] #Yeh woh entry hogi jo accuracy mein dusre number par hai.
            
            if naya_score > purana_score:
                print(f"Naya score is better ({naya_score})")
            else:
                print("ℹ️ Purana model score behtar hai, lekin hum top model ko register kar rahe hain.")
        else:
            print("Yeh phela model e theak tha issi ko registered kro")

        # --- REGISTRATION (Mohar Lagana) ---
        model_name = "Machine_Failure_Model"
        mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name) #Yeh woh run_id ke model ke register kar do.
        print(f"✅ Model '{model_name}' register ho gaya!")

        # --- EXPORT LOGIC (Docker ke liye model nikalna) ---
        # MLflow se pucho ke model ka asli rasta (path) kya hai
        asli_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="random_forest_model")
        
        # Purana folder delete karke naya "best_model" folder banao
        if os.path.exists("best_model"):
            shutil.rmtree("best_model")
        
        shutil.copytree(asli_path, "best_model")
        print("📁 Best model 'best_model' folder mein export ho gaya hai (Docker Ready)!")

    else:
        print("Koi run nahi mili, pehle training file chalao!")

if __name__ == "__main__":
    faisla_sunao()
# Interviewer: "Bhai, register_model kyun kiya? log_model toh pehle hi kar diya tha."
# Aapka Jawab:
# "Sir, log_model sirf ek experiment ka record save karta hai.
#  Lekin register_model usse Model Lifecycle mein daal deta hai. 
# Registry mein model ko 'Version' mil jata hai aur wahan se hum usse Staging ya Production ka tag de sakte hain. Bina register kiye, hamari deployment pipeline ko
#  pata nahi chalega ke kaunsa model 'Best' hai aur kahan se uthana hai."