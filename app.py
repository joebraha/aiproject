import streamlit as st  #Web App

#title
st.title("Sentiment Analysis")


def analyze(input, model):
    return "This is a sample output"

#text insert
input = st.text_area("insert text to be analyzed", value="lorem ipsum...", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
model = st.selectbox("choose a transformer model", ['a', 'b', 'c'], index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

if st.button('Analyze'):
    st.write(analyze(input, model))
else:
    st.write('Goodbye')

