import streamlit as st  #Web App
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


#title
st.title("Sentiment Analysis")


def analyze(input, model):
    return "This is a sample output"


# load my fine-tuned model
fine_tuned = None


#text insert
input = st.text_area("insert text to be analyzed", value="Nice to see you today.", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
option = st.selectbox(
    'Choose a transformer model:',
    ('Default', 'Fine-Tuned' , 'Custom'))


if option == 'Fine-Tuned':
    model = TFAutoModelForSequenceClassification.from_pretrained(fine_tuned)
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
else:
    classifier = pipeline('sentiment-analysis')


if st.button('Analyze'):
    st.write(classifier(input))
else:
    st.write('Excited to analyze!')

