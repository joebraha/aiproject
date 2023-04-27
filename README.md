---
title: Aiproject
emoji: 📊
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.17.0
app_file: app.py
pinned: false
---


# Milestone 4

This project demonstrates the use of the HuggingFace transformers API to fine-tune a text classification language model using a sample dataset. Specifically, the BERT model was trained to identify the "toxicity" of messages based on the training dataset from [kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The model was then deployed to a HuggingFace space using Streamlit.

The project can be viewed [here](https://sites.google.com/nyu.edu/joebraha-aiproject/), or direct on [HuggingFace](huggingface.co/spaces/jbraha/aiproject).

Additionally, a brief demonstration of the project and it's pieces can be viewed [here](https://youtu.be/ONoCm7tfmME).

### Preliminary Note

The space takes quite a long time to load up and operate due to the configuration of streamlit site rendering. A future implementation should look to solve this problem.


## Training
Google Colab was used to power the training of the model. The notebook used for training is available at [`working_training.ipynb`](working_training.ipynb).

### Imports
The following packages were used in the training:
- torch - PyTorch was used to power the training
- transformers - HuggingFace transformers API was used to access the BERT model and perform the training
- pandas - Pandas was used to read the training data from the csv file
- sklearn - Scikit-learn was used to split the training data into training and validation sets

### Setup

The model and tokenier were loaded from the transformers API, and the model set to utilize the GPU for training. The training arguments were set, with the notable properties being 2 training epochs and a batch size of 16.

Two Dataset classes were built from `torch.utils.data.Dataset` to hold the strings (`TokenizerDataset`) and habdle the tokenization (`TokenizedDataset`) of the training data. The `__getitem__` method of `TweetDataset` pulls the strings, tokenizes them, and prepares the encodings for training.

Finally, the `MultilabelTrainer` class was built from the `Trainer` parent class to properly handle the conversion to a six-headed model in the training.

### Load Data

The training data was loaded from the csv file and split into training and validation sets. The data was then split into the strings and the labels, and converted to a `list` as expected by the trainer.

### Prepare Datasets and Trainer

the strings were loaded into the `TokenizerDataset` and then that into the `TweetDataset`. The length of the objects were printed for a quick sanity check, and then the trainer was built with the model, arguments from above, and the data.

### Training

`!nvidia-smi` was run to double check the cpu before beginning the training, and then the trainer was run for approximately 2.5 hours.


## StreamLit App

The interface of the app consists of a text box for input (`st.text_area`), a select box for a model (some pre-trained models, and the fine-tuned model) (`st.selectbox`), an analyze button, and a table displaying the fine-tuned results (`st.table`). 

Each of the models are loaded in the beginning, the pre-trained ones from HuggingFace and the fine-tuned from the local repository. Two functions are used to assist in the processing: `unpack`, which takes the result and turns it into a dictionary with the proper labels from the `labels` dictionary, and `add_to_table`, which pulls out the two labels with the highest score and adds the data as a `list` to the `output` table.

The table is preloaded with some messages, which are processed with the above functions, and then the table is displayed.

When the user presses "Analyze", the appropriate model is used to predict the sentiment. If it's one of the pre-trained, then the output will be displayed as it is wiht `st.write`. If the fine-tuned model was chosen, then the data will be processed and added to the table, and the table display will update automatically.

## Results

Through this processes, the model was successfully trained, and the web app environment successfully deployed. However, as can be seen by the low label values for the sample training set, the model did not learn as effectively as it should have. It would seem that the training was not able to deeply learn after approximately 500 batches, after which the training loss oscillated around .02 to .05 for the rest of the training.
