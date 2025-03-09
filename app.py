import os
import re
import base64
import streamlit as st
from streamlit_navigation_bar import st_navbar

import templates as pg



# Page configuration
st.set_page_config(
    initial_sidebar_state="collapsed",
    layout="wide", 
    page_title = "MMM Scenario Planning"
    )


# WebApp Cosmetics
parent_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(parent_dir, "static/images/logo.svg")

options = {
    "show_menu": True,
    "show_sidebar": False,
}



def css_to_dict():
    """Converts CSS to a Python dictionary."""

    with open('static/css/styles.css') as f:
        styles = f.read()

    css_dict = {}
    rules = re.findall(r'([^{]+){([^}]+)}', styles)

    for selector, declarations in rules:
        selector = selector.strip()
        css_dict[selector] = {}

        for declaration in declarations.split(';'):
            if declaration.strip():
                key, value = declaration.split(':', 1)
                css_dict[selector][key.strip()] = value.strip()

    return css_dict


styles = css_to_dict()



page = st_navbar(
    ['Scenario Planning', 'Budget Minimizing', 'Attendee Maximizing'],
    logo_path=logo_path,
    # urls=urls,
    styles=styles,
    options=options
)



# Background Image
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# =====================================================================================================================
# Background Image
# =====================================================================================================================
img = get_img_as_base64("static/images/soul08.png")

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





# =====================================================================================================================
# Functionalities begin
# =====================================================================================================================
functions = {
    "Home": pg.show_intro,
    "Scenario Planning": pg.show_scenario,
    "Budget Minimizing": pg.show_minimization,
    'Attendee Maximizing': pg.show_maximization,

}


go_to = functions.get(page)
if go_to:
    go_to()
