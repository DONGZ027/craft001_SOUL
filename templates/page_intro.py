import streamlit as st
import pandas as pd
import numpy as np


def show_intro():
    st.title("Hi there!")
    col1, spacing_col, col2 = st.columns([5, 1, 5]) 


    with col1:
        st.write("")
        st.write('''
                Welcome to the SOUL - Scenario and Optimization Unified Lab! 
                
                With access to the latest media mix model results, you can plan and vision the effective media contribution over and over again.  
                ''')
        
    # The middle column creates the white space
    with spacing_col:
        st.write("")

    with col2:
        st.write("")
        st.image("static/images/soul-joe-mr-mittens3.png", width= 350)