#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 04:22:41 2021

@author: swarneil99
"""
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("rfmodel.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def shipment_prediction(warehouse, shipmentmode, cccalls, custratings, cost, priorpurchase, importance, Gender, discount, weight):
    
   
   
    prediction=classifier.predict([[warehouse, shipmentmode, cccalls, custratings, cost, priorpurchase, importance, Gender, discount, weight]])
    print(prediction)
    return prediction



def main():
    st.title("Shipment Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Shipment Analysis ML App </h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    warehouse = st.text_input("Type of Warehouse","Type Here")
    shipmentmode = st.text_input("Type of Shipment Mode","Type Here")
    cccalls = st.text_input("Customer Calls","Type Here")
    custratings = st.text_input("Customer Ratings","Type Here")
    cost = st.text_input("Cost","Type Here")
    priorpurchase = st.text_input("Prior Purchases","Type Here")
    importance = st.text_input("Importance of the Product","Type Here")
    Gender = st.text_input("Gender","Type Here")
    discount = st.text_input("Discount","Type Here")
    weight = st.text_input("Weight","Type Here")
    
    result=""
    if st.button("Predict"):
        result=shipment_prediction(warehouse, shipmentmode, cccalls, custratings, cost, priorpurchase, importance, Gender, discount, weight)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
