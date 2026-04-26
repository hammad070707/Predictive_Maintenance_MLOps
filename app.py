# API (FastAPI) banayenge—jo ek "Khidki" (Window) ki tarah hogi.
# Manager machine ka data (Temperature, Speed) us khidki mein dalega.
# Hamara model andar se check karke batayega: "Machine theek hai" ya "Kharab hone wali hai".
from fastapi import FastAPI #FastAPI ek modern framework hai (jaise Flask hota hai) jo APIs banane ke kaam aata hai. Yeh bohot fast hai, isliye iska naam FastAPI hai.
import mlflow.sklearn #Hamara model MLflow ki registry mein chupa hua hai. Yeh library usse wahan se Khinch kar (Load) bahar nikalne mein madad karegi
import pandas as pd#Model ne training ke waqt "Table" (DataFrame) par seekha tha. Jab hum prediction karenge, toh humein user ke data ko bhi table mein badalna padega.
import uvicorn #FastAPI khud nahi chal sakti, usse chalane ke liye ek Server Engine chahiye hota hai, jiska naam uvicorn hai.
app=FastAPI()#app = FastAPI(): Humne FastAPI ka ek Object (Main App) bana liya. Ab saara kaam is app ke zariye hoga.
model_name="Machine_Failure_Model"#Humne bataya ke humein "Machine_Failure_Model" chahiye.
model_version="latest"#Model ke version number ko bhi specify karna zaroori hai, taki humein pata chale ke kaunsa version use karna hai.
model=mlflow.sklearn.load_model("best_model")#Model ko MLflow registry se load karna. Yeh line model ko humare code mein le aayegi, taki hum usse prediction ke liye use kar sakein.
#models:/: Yeh MLflow ka ek rasta hai jo Model Registry ki taraf jata hai.
@app.get("/")
def home():
    return {"message": "Welcome to the Machine Failure Prediction API!"}#Yeh ek simple route hai jo jab user "/" (Home) pe aayega, toh ek welcome message dikhayega.
from pydantic import BaseModel #Pydantic ek library hai jo data validation ke liye use hoti hai. Isse hum ensure kar sakte hain ki user se aane wala data sahi format mein ho.
class MachineData(BaseModel): #MachineData ek class hai jo BaseModel se inherit karti hai. Is class mein hum define karenge ki user se kaun-kaun se data fields chahiye.
    Type:int
    Air_temperature: float #Yeh woh temperature ke data bhejega (e.g., 300).
    Process_temperature: float #Yeh woh process temperature ke data bhejega (e.g., 350).
    Rotational_speed: float #Yeh woh RPM ke data bhejega (e.g., 1500).
    Torque: float #Yeh woh torque ke data bhejega (e.g., 50).
    Tool_wear: float #Yeh woh tool wear ke data bhejega (e.g., 10).


@app.post("/predict") #@app.post: Hum "POST" method use kar rahe hain kyunki user hamein data Bhej (Send) raha hai.
def predict(input_data:MachineData): #User hamein JSON format mein data bhejega (e.g., Temperature: 300, RPM: 1500). Python usse ek Dictionary (dict) bana kar hamein dega.
#  FastAPI ka "Auto-Translator" (Parsing)
# Jab koi user tumhari API ko JSON bhejta hai, toh woh asal mein sirf ek Text (String) hoti hai jo internet par travel kar rahi hoti hai.
# FastAPI ka kaam: Jaise hi woh JSON text tumhare server par pahonchta hai, FastAPI usse foran (automatically) parh kar ek Python Dictionary (dict) mein badal deta hai.
# Yani tumhare function def predict(data: dict) tak pahonchte-pahonchte, woh JSON bacha hi nahi, woh Dictionary ban chuka hai.
    data_dict=input_data.dict() #Matlab: User ne jo data bheja woh ek Pydantic Object tha. Humne usse ek simple Python Dictionary bana liya taake hum asani se data_dict["Torque"] karke uski value nikal sakein.
    input_df = pd.DataFrame([{
        "Type": data_dict["Type"],
        "Air temperature [K]": data_dict["Air_temperature"],
# "Air temperature [K]": data_dict["Air_temperature"]
# The Logic: Training ke waqt model ne column ka naam parha tha Air temperature [K].
# Ab agar hum usse sirf Air_temperature bhejenge, toh model kahega: "Yeh kaun hai? Maine toh iss naam ka koi column dekha hi nahi!"
# Isliye humne Mapping ki: "Bhai jo User ne 'Air_temperature' bheja hai, usse tum wahi samjho jo tumhare liye 'Air temperature [K]' tha."

        "Process temperature [K]": data_dict["Process_temperature"],
        "Rotational speed [rpm]": data_dict["Rotational_speed"],
        "Torque [Nm]": data_dict["Torque"],
        "Tool wear [min]": data_dict["Tool_wear"]
    }])
    prediction=model.predict(input_df) #Humne model se kaha: "Bhai, yeh lo data (df), ab apna dimaag lagao aur batao result kya hai.
    result="Machine Failure" if prediction[0]==1 else "Machine is Healthy"#Model ne 1 return kiya toh matlab machine kharab hone wali hai, aur agar 0 return kiya toh matlab machine theek hai.
    return {"result":result} #Yeh woh result kiya toh humein JSON format mein data bhejega (e.g., {"result": "Machine Failure"}).
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# host="0.0.0.0": Iska matlab hai yeh API tumhare computer ke har network interface par available hogi (Docker ke liye yeh bohot zaroori hai).
# port=8000: Yeh woh darwaza (Port) hai jahan se log tumhari API se baat karenge.