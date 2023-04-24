import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments



model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)
max_len = 200

training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
    )

# dataset class that inherits from torch.utils.data.Dataset
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.tok = tokenizer
    
    def __getitem__(self, idx):
        # encoding = self.tok(self.encodings[idx], truncation=True, padding="max_length", max_length=max_len)
        item = { key: torch.tensor(val[idx]) for key, val in self.encoding.items() }
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
class TokenizerDataset(Dataset):
    def __init__(self, strings):
        self.strings = strings
    
    def __getitem__(self, idx):
        return self.strings[idx]
    
    def __len__(self):
        return len(self.strings)
    




train_data = pd.read_csv("data/train.csv")
train_text = train_data["comment_text"]
train_labels = train_data[["toxic", "severe_toxic", 
                           "obscene", "threat", 
                           "insult", "identity_hate"]]

test_text = pd.read_csv("data/test.csv")["comment_text"]
test_labels = pd.read_csv("data/test_labels.csv")[[
                           "toxic", "severe_toxic", 
                           "obscene", "threat", 
                           "insult", "identity_hate"]]

# data preprocessing



train_text = train_text.values.tolist()
train_labels = train_labels.values.tolist()
test_text = test_text.values.tolist()
test_labels = test_labels.values.tolist()




# prepare tokenizer and dataset

train_strings = TokenizerDataset(train_text)
test_strings = TokenizerDataset(test_text)

train_dataloader = DataLoader(train_strings, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_strings, batch_size=16, shuffle=True)




# train_encodings = tokenizer.batch_encode_plus(train_text, \
#                             max_length=200, pad_to_max_length=True, \
#                             truncation=True, return_token_type_ids=False \
#                             )
# test_encodings = tokenizer.batch_encode_plus(test_text, \
#                             max_length=200, pad_to_max_length=True, \
#                             truncation=True, return_token_type_ids=False \
#                             )


train_encodings = tokenizer.encode(train_text, truncation=True, padding=True)
test_encodings = tokenizer.encode(test_text, truncation=True, padding=True)


f = open("traintokens.txt", 'a')
f.write(train_encodings)
f.write('\n\n\n\n\n')
f.close()

g = open("testtokens.txt", 'a')
g.write(test_encodings)
g.write('\n\n\n\n\n')

g.close()



# train_dataset = TweetDataset(train_encodings, train_labels)
# test_dataset = TweetDataset(test_encodings, test_labels)





# # training
# trainer = Trainer(
#     model=model, 
#     args=training_args, 
#     train_dataset=train_dataset, 
#     eval_dataset=test_dataset
#     )


# trainer.train()







