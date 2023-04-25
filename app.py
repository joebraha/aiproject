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

def add_to_table(result, output):
    highest = (max(result, key=result.get), result[max(result, key=result.get)])
    result.pop(max(result, key=result.get))
    output.append([input, highest[0], highest[1], max(result, key=result.get), 
                   result[max(result, key=result.get)]])




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
strings = [ "D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)",
            "Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.",
            "Why can't you believe how fat Artie is? Did you see him on his recent appearence on the Tonight Show with Jay Leno? He looks absolutely AWFUL! If I had to put money on it, I'd say that Artie Lange is a can't miss candidate for the 2007 Dead pool!  \
                Kindly keep your malicious fingers off of my above comment, . Everytime you remove it, I will repost it!!!",
            "Thank you. This would make my life complete.  ",
            "Would you both shut up, you don't run wikipedia, especially a stupid kid.", 
            "Please stop. If you continue to vandalize Wikipedia, as you did to Homosexuality, you will be blocked from editing.",
            "== Arabs are committing genocide in Iraq, but no protests in Europe. ==  May Europe also burn in hell.",
            ":yeah, thanks for reviving the tradition of pissing all over articles because you want to live out your ethnic essentialism. Why let mere facts get into the way of enjoying that.",
            "==Category:Italian Jews==  :Category:Italian Jews, which you created, has been nominated for possible deletion, merging, or renaming. If you would like to participate in the discussion, you are invited to add your comments at the category's entry on the Categories for discussion page. Thank you.",
            "KSchwartz is an annoying person who often smells of rotten fish and burnt animal hair.  He has a passing interest in Wikipedia."
            ]

for string in strings:
    result = classifier(string)
    result = result[0]
    result = unpack(result)
    add_to_table(result, output)

st.table(output)


if st.button('Analyze'):
    result = classifier(input)
    result = result[0]
    if option == 'Fine-Tuned':
        result = unpack(result)
        add_to_table(result, output)
        # st.table(output)
    else:
        st.write(result)
else:
    st.write('Excited to analyze!')






