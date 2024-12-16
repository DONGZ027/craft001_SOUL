import streamlit as st
import pandas as pd
import numpy as np


def show_intro():
    st.title("Hi there!")
    col1,  col2 = st.columns(2) 


    with col1:
        st.write("")
        st.write('''
                I am BayMAX - Budgeting Ad Yield Maximizer. Welcome to the planning room for Disneyland Paris.

                Armed with the most recent Media Mix Model results, I can help you plan the advertising budget for the next future years. 
                ''')
        

    with col2:
        st.write("")
        st.image("static/images/DLP_logo.PNG", width= 400)