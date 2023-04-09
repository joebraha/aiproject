import streamlit as st  #Web App
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


#title
st.title("Sentiment Analysis")


def analyze(input, model):
    return "This is a sample output"

#text insert
input = st.text_area("insert text to be analyzed", value="Nice to see you today.", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
model_name = st.text_input("choose a transformer model", value="")
if model_name:
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

else:
    classifier = pipeline('sentiment-analysis')


if st.button('Analyze'):
    st.write(classifier(input))
else:
    st.write('Goodbye')

