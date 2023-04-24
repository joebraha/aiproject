import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6).to(device)
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

    
class TokenizerDataset(Dataset):
    def __init__(self, strings):
        self.strings = strings
    
    def __getitem__(self, idx):
        return self.strings[idx]
    
    def __len__(self):
        return len(self.strings)
    

train_data = pd.read_csv("data/train.csv")
print(train_data)
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

class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.tok = tokenizer
    
    def __getitem__(self, idx):
        print(idx)
        # print(len(self.labels))
        encoding = self.tok(self.encodings.strings[idx], truncation=True, 
                            padding="max_length", max_length=max_len)
        # print(encoding.items())
        item = { key: torch.tensor(val) for key, val in encoding.items() }
        item['labels'] = torch.tensor(self.labels[idx])
        # print(item)
        return item
    
    def __len__(self):
        return len(self.labels)





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

# train_encodings = tokenizer(train_text, truncation=True, padding=True)
# test_encodings = tokenizer(test_text, truncation=True, padding=True)

train_dataset = TweetDataset(train_strings, train_labels)
test_dataset = TweetDataset(test_strings, test_labels)

print(len(train_dataset.labels))
print(len(train_strings))


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


# training
trainer = MultilabelTrainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset
    )

trainer.train()