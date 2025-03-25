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
                
                Media spend planning is an art of tuning and balancing. 
                Equiped with the latest media mix model results and advanced algorithms, this is a place for you to rehearsal over and over again until you reach the perfect harmony across media channels.  
                ''')
        
    # The middle column creates the white space
    with spacing_col:
        st.write("")

    with col2:
        st.write("")
        st.image("static_files/images/soul-joe-mr-mittens_piano2.png", width= 500)