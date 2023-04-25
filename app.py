import streamlit as st  #Web App
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


#title
st.title("Sentiment Analysis")


def analyze(input, model):
    return "This is a sample output"

# load my fine-tuned model
fine_tuned = "res"
labels = {'LABEL_0': 'toxic', 'LABEL_1': 'severe_toxic', 'LABEL_2': 'obscene', 'LABEL_3': 'threat',
          'LABEL_4': 'insult', 'LABEL_5': 'identity_hate'}


# make a dictionary of the labels and values
def unpack(result):
    output = {}
    for res in result:
        output[labels[res['label']]] = res['score']
    return output


#text insert
input = st.text_area("Insert text to be analyzed", value="you stink", 
                     height=None, max_chars=None, key=None, help=None, on_change=None, 
                     args=None, kwargs=None, placeholder=None, disabled=False, 
                     label_visibility="visible")

option = st.selectbox(
    'Choose a transformer model:',
    ('Default', 'Fine-Tuned' , 'Roberta'))


if option == 'Fine-Tuned':
    model = AutoModelForSequenceClassification.from_pretrained(fine_tuned)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, top_k=None)
elif option == 'Roberta':
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
else:
    classifier = pipeline('sentiment-analysis')



output = []
output.append(['Tweet', 'Highest', 'Score', 'Second Highest', 'Score'])

if st.button('Analyze'):
    result = classifier(input)
    result = result[0]
    if option == 'Fine-Tuned':
        result = unpack(result)
        highest = (max(result, key=result.get), result[max(result, key=result.get)])
        result.pop(max(result, key=result.get))
        output.append([input, highest[0], highest[1], max(result, key=result.get), result[max(result, key=result.get)]])
        st.table(output)
    else:
        st.write(result)
else:
    st.write('Excited to analyze!')






