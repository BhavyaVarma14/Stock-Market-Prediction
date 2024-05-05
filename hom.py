import streamlit as st
import sys
import os
user_credentials = {
        'admin': 'admin',
        'bhy': 'admin',
    }

custom_css = """
    <style>
        body {
            background-color: #f0f0f0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

gradient_bg = """
        <style>
        [data-testid="stAppViewContainer"] {
            
            background-image: url("https://images.axios.com/bHCnHHwxalCV9uHEFBCP2KpVO70=/0x0:1920x1080/1920x1080/2019/09/30/1569844465490.gif");
            width:100%;
            height:100%;
            background-size:cover;
            
        
        }  
        [data-testid="stMarkdownContainer"] {
            
           color:#01CF73; 
           font-weight:10000px;
           } 
        </style>
    """
st.markdown(gradient_bg, unsafe_allow_html=True)

st.markdown("<h1  style='color:#01CF73; font-size: 50px; font-weight: bold;'>Stock Price Prediction Homepage</h1>", unsafe_allow_html=True)

st.write("<p  style='color:#000006; font-size: 30px; font-weight: bold; text-align:center; '>LOGIN</p>", unsafe_allow_html=True)

    # Customize the layout and style using Streamlit native features
with st.form("login_form"):                                                                                                                                                                                                                                                                                                                                                             
        st.write("<h5 style='color : #1a1716; font-weight:bold;'>Username</h5>",unsafe_allow_html=True)
        username = st.text_input("")
        st.write("<h5 style='color : #1a1716; font-weight:bold;'>Password</h5>",unsafe_allow_html=True)
        password = st.text_input(" ", type="password")
        #login_button = st.form_submit_button("Login")
        if st.form_submit_button("Login"):
            if username in user_credentials:
                 if user_credentials[username] == password:
        
                    os.system(f"python -m streamlit run app2.py")
            else:
                st.error("Invalid username. Please try again.")




    # Button to launch the stock price prediction app
