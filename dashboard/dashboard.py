# Import modules 
import os
import numpy as np
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import lightgbm as lgb

########################################
abs_path = os.path.dirname(os.path.realpath(__file__))
html_header="""
<head>
<title>PHomeCredit</title>
<meta charset="utf-8">
<meta name="keywords" content="home credit risk, dashboard, Hanen Ben Brahim">
<meta name="description" content="Home Credit Risk Dashboard">
<meta name="author" content="Hanen BEN BRAHIM">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<h1 style="font-size:300%; color:#838383; font-family:Georgia"> Home Credit Default Risk Dashboard <br>
 <h2 style="font-size:200%; "color:#BFBCBC; font-family:Georgia"> Hanen Ben Brahim </h2> <br></h1>
"""

st.set_page_config(page_title="Home Credit Default Risk Dashboard", page_icon="", layout="wide")


with st.container():
  col1, col2, col3, col4 = st.columns([1,26,1,5])
  with col1:
    st.write("")
  with col2:
    st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
    st.markdown(html_header, unsafe_allow_html=True)
    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
  with col3:
    st.write("")
  with col4:
    st.write("")  

html_line="""
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 1.5px;"></p>
"""
st.markdown(html_line, unsafe_allow_html=True)


html_card_header1="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 350px;
   height: 50px;">
    <h3 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;"> ID Client </h3>
  </div>
</div>
"""
html_card_footer1="""
<div class="card">
  <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 350px;
   height: 50px;">
    <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;"> </p>
  </div>
</div>
"""
html_card_header2="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 350px;
   height: 50px;">
    <h3 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Probabilité</h3>
  </div>
</div>
"""
html_card_footer2="""
<div class="card">
  <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 350px;
   height: 50px;">
    <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;"> Seuil : THRESHOLD</p>
  </div>
</div>
"""
html_card_header3="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 350px;
   height: 50px;">
    <h3 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Décision : VALEUR</h3>
  </div>
</div>
"""
html_card_footer3="""
<div class="card">
  <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 350px;
   height: 50px;">
    <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Score : VALEUR</p>
  </div>
</div>
"""
### Block 0 #########################################################################################

# Spécifications du dashboard
# 1- Permettre de visualiser le score et l’interprétation de ce score pour chaque 
#    client de façon intelligible pour une personne non experte en data science.
# 2- Permettre de visualiser des informations descriptives relatives à un client
#    (via un système de filtre).
# 3- Permettre de comparer les informations descriptives relatives à un client à 
#    l’ensemble des clients ou à un groupe de clients similaires.

# URL de l'API
api_adress = "https://hanen-p7-oc-ben.herokuapp.com/"

# Liste des IDs des clients
@st.cache
def get_id_list():
    response = requests.get(api_adress + "get_id_list/")
    content = json.loads(response.content)
    id_list = content['id_list']
    return id_list

# Résultats de prédiction pour un client 
@st.cache
def get_score(id):
    response = requests.get(api_adress + "get_score/?id=" + str(id))
    content = json.loads(response.content)
    score = content['score']
    proba = content['proba']
    thresh = content['thresh']
    return score, proba, thresh

# Les informations descriptives relatives à un client
@st.cache
def get_information_descriptive(id):
    response = requests.get(api_adress + "get_information_descriptive/?id=" + str(id))
    content = json.loads(response.content)
    data_client = pd.read_json(content['X'])
    return data_client

# Les informations descriptives relatives à l'ensemble de clients 

@st.cache
def get_data():
    response = requests.get(api_adress + "get_data/")
    content = json.loads(response.content)
    X_tr_proc = pd.read_json(content['X'])
    return X_tr_proc

# # Liste de feature importance 
# @st.cache
# def get_features_importances():
#     response = requests.get(api_adress + "get_feature_importance/")
#     content = json.loads(response.content)
#     features_importances = pd.read_json(content['features_importances'])
#     return features_importances



### Block 1#########################################################################################
liste_id = get_id_list()
data = get_data() 


df_sans_id = data.drop(columns=['identifiant'])
temp_lst = df_sans_id.columns.to_list()

cat_features = df_sans_id.select_dtypes(exclude=[np.number]).columns.to_list()
num_features = df_sans_id.select_dtypes(include=[np.number]).columns.to_list()
# features_importances = get_features_importances()


with st.expander("Mission du dashboard"):
  st.write("Dashboard pour visualiser les informations sur un client demandant un \
            crédit bancaire et le comparer avec des profils similaires de client")


with st.container():
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1,15,1,15,1,15,1])
    with col1:
        st.write("")
    with col2:
        st.markdown(html_card_header1, unsafe_allow_html=True)
        selected_id = st.selectbox('Veuillez saisir un ID client :', liste_id)
        df_client = get_information_descriptive(selected_id)

    with col3:
        st.write("")
    with col4:
        st.markdown(html_card_header2, unsafe_allow_html=True)
        score , proba, threshold = get_score(selected_id)
        fig_c2 = go.Figure(go.Indicator(
            mode="number",
            value= round(proba, 2),
            number={'suffix': "", "font": {"size": 40, 'color': "#A7A7A7", 'family': "Arial"}},
            domain={'x': [0, 1], 'y': [0, 1]}))
        fig_c2.update_layout(autosize=False,
                             width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                             paper_bgcolor="#fbfff0", font={'size': 20})
        st.plotly_chart(fig_c2)
        st.markdown(html_card_footer2.replace("THRESHOLD", str(round(threshold, 2)), 1), unsafe_allow_html=True)
    with col5:
        st.write("")
    with col6:
        color = "#7CFC00" if score==0 else  "#FF0000"
        TEXT = "Crédit accordé" if score==0 else "Crédit refusé"
        st.markdown(html_card_header3.replace("VALEUR", TEXT, 1), unsafe_allow_html=True)
        fig_c3 = go.Figure(go.Indicator(
        mode = "gauge",
        gauge = {'shape': "bullet", 
                 'axis': {'visible':False},
                 'bgcolor':color},
        domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
        ))
        fig_c3.update_layout(autosize=True,
                             width=350, height=90, margin=dict(l=20, r=20, b=20, t=30),
                             paper_bgcolor="#fbfff0", font={'size': 20})                  
        st.plotly_chart(fig_c3)
        st.markdown(html_card_footer3.replace("VALEUR", str(int(score)), 1), unsafe_allow_html=True)

    with col7:
        st.write("")

html_br="""
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)


html_card_header4="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 850px;
   height: 50px;">
    <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Données client</h4>
  </div>
</div>
"""

### Block 2#########################################################################################
with st.container():
    col1, col2, col3 = st.columns([1,42,1])
    with col1:
        st.write("")

    with col2:
        # Données des clients
          st.markdown(html_card_header4, unsafe_allow_html=True)

    with col3:
        st.write("")


html_br="""
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)


if st.checkbox("Afficher les informations descriptives du client"):
  with st.container():
      col1, col2, col3 = st.columns([1,42,1])
      with col1:
          st.write("")

      with col2:
          st.dataframe(df_client.set_index('identifiant'))

      with col3:
          st.write("")



html_card_header5="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 850px;
   height: 50px;">
    <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Jeu de Données</h4>
  </div>
</div>
"""
### Block 3 #########################################################################################
with st.container():
    col1, col2, col3 = st.columns([1,42,1])
    with col1:
        st.write("")

    with col2:
        # Données des clients
          st.markdown(html_card_header5, unsafe_allow_html=True)

    with col3:
        st.write("")


html_br="""
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)


 ### Block 4 #########################################################################################
if st.checkbox("Afficher les informations descriptives de l'ensemble des clients"):
 
  with st.container():
      col1, col2, col3 = st.columns([1,42,1])
      with col1:
          st.write("")

      with col2:
          # Données des clients existant dans le jeu 
          variable = st.checkbox("Customer's data")
          # st.markdown(html_card_header15, unsafe_allow_html=True)
          st.dataframe(data.set_index('identifiant'))
           

      with col3:
          st.write("")

  html_br="""
  <br>
  """
  st.markdown(html_br, unsafe_allow_html=True)


  ### Block 5 #########################################################################################

  html_card_header6="""
  <div class="card">
    <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 250px;
    height: 50px;">
      <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Analyse Univariée</h4>
    </div>
  </div>
  """
  html_card_footer6="""
  <div class="card">
    <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #9C9B9B; padding-top: 1rem;; width: 250px;
    height: 50px;">
      <p class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 0px 0;">Montly Value</p>
    </div>
  </div>
  """

  ### Analyse Univariée ###
  with st.container():
    col1, col2, col3, col4, col5 = st.columns([1,15,2,15,1])
    with col1:
      st.write("")
    with col2:
      st.markdown(html_card_header6, unsafe_allow_html=True)
      variable = st.selectbox ("Quel attribut voulez-vous analyser?", 
                                                temp_lst,
                                                )
      fig = px.histogram(data,
                        x=variable,
                        title= 'Distribution de la variable : ' + variable,
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
      categories = st.multiselect("Sélectionnez les variables à comparer : ", 
                                  options=data.columns.to_list(),
                                  default= columns_lst[:5],
                                  )
      # Choisir les 5 premieères variables sélectionées 
      if len(categories)<5:
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
            name='Ensemble de clients'
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

  html_br="""
  <br>
  """
  st.markdown(html_br, unsafe_allow_html=True)


### Block 6 #########################################################################################
html_card_header8="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 850px;
  height: 50px;">
    <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Interprétation des résultats</h4>
  </div>
</div>
"""
with st.container():
    col1, col2, col3 = st.columns([1,42,1])
    with col1:
        st.write("")

    with col2:
            st.markdown(html_card_header8, unsafe_allow_html=True)

    with col3:
        st.write("")

html_br="""
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)
html_card_header9="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #9C9B9B; padding-top: 5px; width: 450px;
  height: 50px;">
    <h4 class="card-title" style="background-color:#9C9B9B; color:#F2EBEB; font-family:Georgia; text-align: center; padding: 10px 0;">Importance de Variables</h4>
  </div>
</div>
"""


if st.checkbox("Afficher l'interprétation des résultats"):
  ### Block 7 #########################################################################################
  with st.container():
    col1, col2, col3 = st.columns([1,42,1])
    with col1:
        st.write("")

    with col2:
            st.markdown(html_card_header9, unsafe_allow_html=True)

    with col3:
        st.write("")

  html_br="""
  <br>
  """
  st.markdown(html_br, unsafe_allow_html=True)
  if st.checkbox("Afficher Importance de Variables"):
    with st.container():
      col1, col3 = st.columns([1,32])
      with col1:
        st.write("")
#       with col2:
#         # features_importances
#         features_importances_df = features_importances.reset_index()
#         features_importances_df.columns = ['Feature', 'Top Features Importance']
#         fig = px.bar(features_importances_df,
#                     x='Top Features Importance',
#                     y='Feature',
#                     )
#         fig.update_layout(height=900)
#         st.plotly_chart(fig)
      with col3:
        st.write("")

      html_br="""
      <br>
      """
      st.markdown(html_br, unsafe_allow_html=True)


  with st.container():
    col1, col2, col3 = st.columns([1,42,1])
    with col1:
        st.write("")

    with col2:
            st.markdown(html_card_header10, unsafe_allow_html=True)

    with col3:
        st.write("")

  html_br="""
  <br>
  """
  st.markdown(html_br, unsafe_allow_html=True)  
  
html_line="""
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
