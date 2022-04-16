# 1. Library imports
import pandas as pd
import pickle
import os
from os import path as op
import fastapi
import uvicorn ## ASGI
import lightgbm as lgb
import shap
import json

# 2. Create app
app = fastapi.FastAPI()
userid: int
 
# 3. API greeting message
@app.get('/')
def index():
    return ("API for Home Credit Default Risk Prediction, created by Hanen Ben Brahim"), 200  
  
# 4. Import dataset  
abs_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(abs_path, 'model', 'light_gbm_f2.sav')
# with open(path, 'rb') as file:
model_obj = pickle.load(open(path, 'rb'))
model = model_obj    
df = pd.read_csv(os.path.join(abs_path,'data', 'sampled_data (2).csv'))

# 5. Get client's ID list
@app.get('/get_id_list/')
def get_id_list():
    temp = sorted(df['identifiant'].values)
    temp_int = [int(i) for i in temp]
    id_list = json.loads(json.dumps(temp_int))
    return jsonify({'status': 'ok',
    		        'id_list': id_list}), 200

# 6. Make a prediction based on the user-entered id
# Returns the predicted class with its respective probability
@app.post('/predict/')
def predict_score(userid):
    int_id = int(userid)
    # load the model from disk
    # model = pickle.load(open(op.join(r'C:\Users\khale\Desktop\projet 7\light_gbm_f2.sav'), 'rb'))
    data_in = df[df['identifiant'] == int_id]
    del (data_in['identifiant'])
    prediction = model.predict(data_in)
    probability = model.predict_proba(data_in).max()
    # Explainability
#     explainer = shap.TreeExplainer(model)
#     shap_value = explainer.shap_values(data_in)
    pos = [108, 109, 130, 126, 66]
    colname = df.columns[pos]
    shap_important_features = colname
    if prediction == 0:
        print('Your loan is accepted')
    elif prediction == 1:
        print('Your loan is not accepted')

    return {
            'prediction': prediction[0],
            'probability': probability,
#             'explainer': explainer.expected_value[1],
#             'shap_val': shap_value[1][0].tolist(),
            'col': shap_important_features.tolist()
            }

   
# 7. Get client informations 
@app.post('/get_descriptives_informations/')
def get_desscriptives_informations(userid):
    df = pd.read_csv(os.path.join(abs_path,'data', 'sampled_data (2).csv'))
    id = int(userid)
    temp_df = df[df['identifiant']==id]
    df_json = temp_df.to_json()
    return ({'df': df_json})
   
@app.get('/get_data/')
def get_data():
    df_json = df.to_json()
    return jsonify({'status': 'ok',
    				'X': df_json}), 200
                    
# 8. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app)
