import os
import base64

import streamlit as st
from streamlit_navigation_bar import st_navbar

  
import pages as pg



#=========================================================================================================
# Page configuration
#=========================================================================================================
st.set_page_config(
    initial_sidebar_state="collapsed",
    layout="wide", 
    page_title = "MMM Scenario Planning"
    )



#=========================================================================================================
# Web App Costmetics
#=========================================================================================================
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "logo.svg")







# styles = {
#     "nav": {   
#         "background-color": "#632323",  #9A463D
#         "justify-content": "left",
#         "font-size": "30px",
#         "font-weight": "bold",
#         "width": "100%",
#         "padding": "0px 0",  # Increase padding to increase height
#     },

#     "img": {
#         "padding-right": "14px",
#         "height": "30px",  # Increase the height of the logo
#     },

#     "span": {
#         "color": "white",
#         "padding": "2px",
#         "font-size": "30px",
#         "font-weight": "bold",
#     },

#     "active": {
#         "background-color": "black",
#         "color": "white",   #var(--text-color)
#         "font-weight": "normal",
#         "padding": "2px",
#         "font-size": "30px",
#         "font-weight": "bold",
#     },
#     "a": {
#         "display": "inline-block",
#         "min-width": "250px",  # Set a minimum width for navigation items
#         "padding": "2px 2px",  # Increase padding to make items wider
#     }
# }



styles = {
    "nav": {
        "background-color": "#632323",
    },
    "div": {
        "max-width": "90rem",
    },
    "span": {
        "color": "white",
        "font-weight": "bold",
        "font-size": "32px",
        "border-radius": "0.5rem",
        "padding": "0.4375rem 0.625rem",
        "margin": "0 0.125rem",
    },
    "active": {
        "background-color": "black",
    },
    "hover": {
        "background-color": "black",
    },
}







options = {
    "show_menu": True,
    "show_sidebar": False,
}




page = st_navbar(
    ['Scenario Planning', 'Attendee Maximization', 'Cost Minimization'],
    logo_path=logo_path,
    # urls=urls,
    styles=styles,
    options=options
)



#=========================================================================================================
# Background Image
#=========================================================================================================


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode() 

img = get_img_as_base64("baymax2_40pct.png") 



page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img}");
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
}}

h1 {{
    font-size: 50px !important;
}}
p {{
    font-size: 32px !important;
}}

/* Ensure navigation bar spans full width */
nav {{
    width: 100%;
    display: flex;
    justify-content: center;
}}

nav a {{
    flex-grow: 1;
    text-align: center;
}}
</style>
"""




st.markdown(page_bg_img, unsafe_allow_html=True)






#=========================================================================================================
# Functionalities begin
#=========================================================================================================
functions = {
    "Home": pg.show_home,
    "Scenario Planning": pg.show_sandbox,
    'Attendee Maximization': pg.show_maximization,
    "Cost Minimization": pg.show_minimization
}


go_to = functions.get(page)
if go_to:
    go_to()






















# h1, h2, h3, h4, h5, h6, p, div, span, li, ul, ol, a {{
#     font-size: 20px !important;
# }}