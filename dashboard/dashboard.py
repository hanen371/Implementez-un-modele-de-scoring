# Import of modules
import os
# import numpy as np
import pandas as pd
import requests
import json
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# import streamlit.components.v1 as components

# let’s add a bit more descriptive text to our UI
st.write("""
# Welcome on this dashboard !
# Context
The company "Ready to distribute" wishes to set up a "credit scoring" tool to calculate the \
probability that a customer will repay his credit, then classify the request as granted or \
refused credit
The label is a binary variable, 0 (will repay the loan on time), 1 (will have difficulty repaying \
the loan)
# Objectives
 1. Create a classification model that will automatically predict the likelihood that a customer \
  can or cannot repay their loan.
 2. Build an interactive dashboard for customer relationship managers to interpret the predictions\
  made by the model, and improve the customer knowledge of customer relationship managers.
# How to use it ?
To predict the score of a specific client, you have to choose the client ID.
To better understand the score, you can compare some informations of the client versus the values of all the others clients.
The multiselect bow allows you to chose which features to compare.
""")
########################################
abs_path = os.path.dirname(os.path.realpath(__file__))
html_header = """
<head>
<title>PHomeCredit</title>
<meta charset="utf-8">
<meta name="keywords" content="home credit risk, dashboard, Hanen Ben Brahim">
<meta name="description" content="Home Credit Risk Dashboard">
<meta name="author" content="Hanen Ben Brahim">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<h1 style="font-size:300%; color:#838383; font-family:Georgia"> Home Credit Default Risk Dashboard <br>
 <h2 style="font-size:200%; "color:#BFBCBC; font-family:Georgia"> Hanen Ben Brahim </h2> <br></h1>
"""

# st.set_page_config(page_title="Home Credit Default Risk Dashboard", page_icon="", layout="wide")
with st.container():
    col1, col2, col3, col4 = st.columns([1, 26, 1, 5])
    with col1:
        st.write("")
    with col2:
        st.markdown('<style>body{background-color: #fbfff0}</style>', unsafe_allow_html=True)
        st.markdown(html_header, unsafe_allow_html=True)
        st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
    with col3:
        st.write("")
    with col4:
        st.write("")

html_line = """
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;"></p>
"""
st.markdown(html_line, unsafe_allow_html=True)

html_card_header1 = """
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 350px;
   height: 50px;">
    <h3 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;"> ID Client </h3>
  </div>
</div>
"""
html_card_footer1 = """
<div class="card">
  <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 350px;
   height: 50px;">
    <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;"> </p>
  </div>
</div>
"""
html_card_header2 = """
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 350px;
   height: 50px;">
    <h3 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Probabilité</h3>
  </div>
</div>
"""

html_card_header3 = """
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 350px;
   height: 50px;">
    <h3 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Décision : VALEUR</h3>
  </div>
</div>
"""
html_card_footer3 = """
<div class="card">
  <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 350px;
   height: 50px;">
    <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Score : VALEUR</p>
  </div>
</div>
"""
### Block 0 #########################################################################################

# URL de l'API
api_adress = "https://hanen-22-benbrahim.herokuapp.com/"
data =  pd.read_csv(os.path.join(abs_path,'data', 'sampled_data (2).csv'))

# # List of customer IDs
# @st.cache
# def get_id_list(id):
#     response = requests.get(api_adress + "/{get_id_list}/?id_list=1")
#     content = json.loads(response.content)
#     id_list = content['id_list']
#     return id_list


# Prediction results for a client
@st.cache
def predict_score(id):
    response = requests.post(api_adress + "predict?userid=" + str(userid))
    content = json.loads(response.content.decode('utf-8'))
    score = content['prediction']
    proba = content['probability']
    return score, proba

data = pd.read_csv(os.path.join(abs_path,'data', 'sampled_data (2).csv'))
# Descriptive information relating to a customer
@st.cache
def get_descriptives_informations(userid):
    response = requests.get(api_adress + "get_descriptives_informations/?userid=" +str(userid))
    content = json.loads(response.content.decode('utf-8'))
    data_client = pd.read_json(content['df'])
    return data_client


# Descriptive information about the set of customers


@st.cache
def get_data():
    response = requests.get(api_adress + "get_data/")
    content = json.loads(response.content)
    # X_ = pd.read_json(content['X'])
    return content


### Block 1#########################################################################################
# liste_id = get_id_list(id)
data = get_data()

df = pd.read_csv(os.path.join(abs_path,'data', 'sampled_data (2).csv'))
df_sans_id = df.drop(columns=['identifiant'])
temp_lst = df_sans_id.columns.to_list()

with st.expander("Dashboard mission"):
    st.write("Dashboard to view information about a customer requesting a \
            bank credit and compare it with similar customer profiles")

with st.container():
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 15, 1, 15, 1, 15, 1])
    with col1:
        st.write("")
    with col2:
        st.markdown(html_card_header1, unsafe_allow_html=True)
        selected_id = st.number_input(label=('Please enter a customer ID :'), min_value=100000)
        df_client = get_descriptives_informations(selected_id)

    with col3:
        st.write("")
    with col4:
        st.markdown(html_card_header2, unsafe_allow_html=True)
        score, proba = predict_score(selected_id)
        fig_c2 = go.Figure(go.Indicator(
            mode="number",
            value=round(proba, 2),
            number={'suffix': "", "font": {"size": 40, 'color': "#A7A7A7", 'family': "Arial"}},
            domain={'x': [0, 1], 'y': [0, 1]}))
        fig_c2.update_layout(autosize=False,
                             width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                             paper_bgcolor="#fbfff0", font={'size': 20})
        st.plotly_chart(fig_c2)
        st.markdown(html_card_footer2.replace("THRESHOLD", str(round(0.5, 2)), 1), unsafe_allow_html=True)
    with col5:
        st.write("")
    with col6:
        color = "#7CFC00" if score == 0 else "#FF0000"
        TEXT = "Crédit accordé" if score == 0 else "Crédit refusé"
        st.markdown(html_card_header3.replace("VALEUR", TEXT, 1), unsafe_allow_html=True)
        fig_c3 = go.Figure(go.Indicator(
            mode="gauge",
            gauge={'shape': "bullet",
                   'axis': {'visible': False},
                   'bgcolor': color},
            domain={'x': [0.1, 1], 'y': [0.2, 0.9]},
        ))
        fig_c3.update_layout(autosize=True,
                             width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                             paper_bgcolor="#fbfff0", font={'size': 20})
        st.plotly_chart(fig_c3)
        st.markdown(html_card_footer3.replace("VALEUR", str(int(score)), 1), unsafe_allow_html=True)

    with col7:
        st.write("")

html_br = """
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)

html_card_header4 = """
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 850px;
   height: 50px;">
    <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Customer data</h4>
  </div>
</div>
"""

### Block 2#########################################################################################
with st.container():
    col1, col2, col3 = st.columns([1, 42, 1])
    with col1:
        st.write("")

    with col2:
        # Customer data
        st.markdown(html_card_header4, unsafe_allow_html=True)

    with col3:
        st.write("")

html_br = """
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)

if st.checkbox("View customer descriptive information"):
    with st.container():
        col1, col2, col3 = st.columns([1, 42, 1])
        with col1:
            st.write("")

        with col2:
            st.dataframe(df_client.set_index('identifiant'))

        with col3:
            st.write("")

html_card_header5 = """
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 850px;
   height: 50px;">
    <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Dataset</h4>
  </div>
</div>
"""
### Block 3 #########################################################################################
with st.container():
    col1, col2, col3 = st.columns([1, 42, 1])
    with col1:
        st.write("")

    with col2:
        # Customer data
        st.markdown(html_card_header5, unsafe_allow_html=True)

    with col3:
        st.write("")

html_br = """
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)

### Block 4 #########################################################################################
if st.checkbox("View descriptive information for all customers"):

    with st.container():
        col1, col2, col3 = st.columns([1, 42, 1])
        with col1:
            st.write("")

        with col2:
            # Data of existing customers in the game
            variable = st.checkbox("Customer's data")
            st.dataframe(data.set_index('identifiant'))

        with col3:
            st.write("")

    html_br = """
  <br>
  """
    st.markdown(html_br, unsafe_allow_html=True)

    ### Block 5 #########################################################################################

    html_card_header6 = """
  <div class="card">
    <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 250px;
    height: 50px;">
      <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Univariate Analysis</h4>
    </div>
  </div>
  """
    html_card_footer6 = """
  <div class="card">
    <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 250px;
    height: 50px;">
      <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Montly Value</p>
    </div>
  </div>
  """

    html_card_header7 = """
  <div class="card">
    <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 550px;
    height: 50px;">
      <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Comparaison avec profils de clients similaires</h4>
    </div>
  </div>
  """

    html_card_footer7 = """
  <div class="card">
    <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 250px;
    height: 50px;">
      <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Montly Value</p>
    </div>
  </div>
  """
    ### Univariate Analysis ###
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([1, 15, 2, 15, 1])
        with col1:
            st.write("")
        with col2:
            st.markdown(html_card_header6, unsafe_allow_html=True)
            variable = st.selectbox("Quel attribut voulez-vous analyser?",
                                    temp_lst,
                                    )
            fig = px.histogram(data,
                               x=variable,
                               title='Distribution de la variable : ' + variable,
                               )

            fig.add_vline(x=df_client[variable].values[0],
                          line_width=3,
                          line_dash="dash")
            fig.update_layout(width=600)
            st.plotly_chart(fig)

        with col3:
            st.write("")
        ### Radar plot ###
        with col4:
            st.markdown(html_card_header7, unsafe_allow_html=True)
            columns_lst = data.columns.to_list()
            categories = st.multiselect("Select the variables to compare: ",
                                        options=data.columns.to_list(),
                                        default=columns_lst[:5],
                                        )
            # Choose the first 5 selected variables
            if len(categories) < 5:
                categories = columns_lst[0:5]
            fig = go.Figure()
            var_client_0 = df_client[categories[0]].mean()
            var_client_1 = df_client[categories[1]].mean()
            var_client_2 = df_client[categories[2]].mean()
            var_client_3 = df_client[categories[3]].mean()
            var_client_4 = df_client[categories[4]].mean()

            fig.add_trace(go.Scatterpolar(
                r=[var_client_0,
                   var_client_1,
                   var_client_2,
                   var_client_3,
                   var_client_4],
                theta=categories,
                fill='toself',
                name='Profil client'
            ))

            var_data_0 = data[categories[0]].mean()
            var_data_1 = data[categories[1]].mean()
            var_data_2 = data[categories[2]].mean()
            var_data_3 = data[categories[3]].mean()
            var_data_4 = data[categories[4]].mean()
            fig.add_trace(go.Scatterpolar(
                r=[var_data_0,
                   var_data_1,
                   var_data_2,
                   var_data_3,
                   var_data_4],
                theta=categories,
                fill='toself',
                name='Set of customers'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )),
                showlegend=False
            )
            fig.update_layout(width=600)
            st.plotly_chart(fig)

        with col5:
            st.write("")

    html_br = """
  <br>
  """
    st.markdown(html_br, unsafe_allow_html=True)

### Block 6 #########################################################################################
html_card_header8 = """
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 850px;
  height: 50px;">
    <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Results interpretation</h4>
  </div>
</div>
"""
with st.container():
    col1, col2, col3 = st.columns([1, 42, 1])
    with col1:
        st.write("")

    with col2:
        st.markdown(html_card_header8, unsafe_allow_html=True)

    with col3:
        st.write("")

html_br = """
<br>
"""

html_line = """
<br>
<br>
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;">
<p style="color:Gainsboro; text-align: right;">By: hanene_benbrahim@yahoo.com</p>
"""

st.markdown(html_line, unsafe_allow_html=True)
