import streamlit as st
import re
import joblib
import pandas as pd
import numpy as np
import st_button

def mycleaning(doc):
   return re.sub("[^a-zA-Z ]","",doc).lower()

model=joblib.load("sentiment_model.pkl")

st.set_page_config(layout="wide")
import streamlit as st
import streamlit.components.v1 as components

# 🕒 Live Clock (JavaScript properly runs here)
components.html("""
    <div style="
        background: linear-gradient(90deg, #ff512f, #dd2476);
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 12px;
    ">
        🕒 <span id="clock"></span>
    </div>

    <script>
        function updateClock() {
            var now = new Date();
            var options = {
                weekday: 'long', year: 'numeric', month: 'long',
                day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
            };
            document.getElementById("clock").innerHTML = now.toLocaleString('en-IN', options);
        }
        setInterval(updateClock, 1000);
        updateClock();
    </script>
""", height=100)

# 🌈 Header
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #43cea2, #185a9d);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        font-size: 44px;
        font-weight: bold;
    ">
        🚀 Sentimental Analysis
    </div>
""", unsafe_allow_html=True)

st.sidebar.image("wallpaperflare.com_wallpaper.jpg") 
st.sidebar.title("About Us📝")
st.sidebar.write("We are a group of AI Engineer at Ducate")

st.sidebar.title("Contact Us ☎️")
st.sidebar.write("1234567890")

st.sidebar.title("About Project 📚")
st.sidebar.write("Restaurant food Review postive or negative")

st.write("\n")
st.write("## Enter Review:")
sample=st.text_input("")

if st.button("Predict"):
      pred=model.predict([sample])
      prob=model.predict_proba([sample])

      if pred[0]==0:
        st.write("### Negative👎")
        st.write(f"#### Confidence score:{prob[0][0]:.2f}")
      else:
        st.write("### Positive😊👍")
        st.write(f"#### Confidence score:{prob[0][1]:.2f}")
        st.balloons()
        
st.write("\n")
st.write("## Bulk Prediction:")
files=st.file_uploader("Select File",type=["csv","txt"])
if files:
    df=pd.read_csv(files,names=["Review"])
    placeholder=st.empty()
    placeholder.dataframe(df)
    if st.button("predict",key="b2"):
      corpus=df.Review
      pred=model.predict(corpus)
      prob=np.max(model.predict_proba(corpus),axis=1)
      df["Sentiment"]=pred
      df["Confidance"]=prob
      df["Sentiment"]=df["Sentiment"].map({0:'Negative👎',1:'Positive😊👍'})
      placeholder.dataframe(df)
    