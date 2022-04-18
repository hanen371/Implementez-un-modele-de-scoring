# import packages
import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request
import json
import lightgbm as lgb
###############################################################
# Load
abs_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(abs_path, 'model', 'light_gbm_f2.sav')
# with open(path, 'rb') as file:
model_obj = pickle.load(open(path, 'rb'))
model = model_obj
thresh = 0.5
X = pd.read_csv(os.path.join(abs_path, 'data', 'sampled_data (2).csv'))

df = pd.read_csv(os.path.join(abs_path,'data', 'df_red.csv'))
y_train_df = df.pop('TARGET')
X = pd.read_csv(os.path.join(abs_path, 'data', 'X_valid_red.csv'))
y_train = pd.read_csv(os.path.join(abs_path, 'data', 'y_valid_red.csv'))
###############################################################
# initiate Flask app
app = Flask(__name__)


# API greeting message
@app.route("/")
def index():
    return ("API for Home Credit Default Risk Prediction, created by Hanen Ben Brahim"), 200


# Get client's ID list
@app.route('/get_id_list/')
def get_id_list():
    temp = sorted(X['identifiant'].values)
    temp_int = [int(i) for i in temp]
    id_list = json.loads(json.dumps(temp_int))
    return jsonify({'status': 'ok',
    		        'id_list': id_list}), 200


# Get client score
@app.route('/get_score/')
def get_score():
    id = int(request.args.get('id'))
    temp_df = X[X['identifiant'] == id]

    del (temp_df['identifiant'])

    proba = model.predict_proba(temp_df).max()
    score = model.predict(temp_df)
    return jsonify({'status': 'ok',
    		        'identifiant': int(id),
                    'score': float(score),
    		        'proba': float(proba),
                    'thresh': float(thresh)}), 200


# Get client informations
@app.route('/get_information_descriptive/')
def get_information_descriptive():
    id = int(request.args.get('id'))
    temp_df = X[X['identifiant'] == id]
    X_df_json = temp_df.to_json()
    return jsonify({'status': 'ok',
    				'X': X_df_json}), 200


@app.route('/get_data/')
def get_data():
    df_json = X.to_json()
    return jsonify({'status': 'ok',
    				'X': df_json}), 200


# Get model feature importance
@app.route('/get_feature_importance/')
def get_feature_importance():
    features_importances = pd.Series(model.feature_importances_,
                                     index=X.drop(columns=['identifiant'])
                                     .columns).sort_values(ascending=False).to_json()
    return jsonify({'status': 'ok',
    		        'features_importances': features_importances}), 200

###############################################################


# main function
if __name__ == "__main__":

    app.run(debug=True)
