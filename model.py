import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import string
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn

# Load datasets
summaries_train = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv')
prompts_train = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv')

# Data preprocessing
summaries_train['text'] = summaries_train['text'].str.lower()
summaries_train['text'] = summaries_train['text'].str.translate(str.maketrans('', '', string.punctuation))

# Lemmatization function
nlp = spacy.load("en_core_web_sm")
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

summaries_train['lemmatized_text'] = summaries_train['text'].apply(lemmatize_text)

# Load pre-trained BERT model and tokenizer
MODEL_PATH = '/kaggle/input/latest-dataset/bert_model'
TOKENIZER_PATH = '/kaggle/input/latest-dataset/bert_tokenizer'
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# Tokenization function
MAX_LEN = 256
def tokenize_for_bert(data, max_len):
    input_ids = []
    attention_masks = []
    for text in data:
        encoded_text = tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=max_len, 
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])
    return torch.tensor(input_ids), torch.tensor(attention_masks)

input_ids, attention_masks = tokenize_for_bert(summaries_train['lemmatized_text'], MAX_LEN)

# Create custom Dataset
class SummarizationDataset(Dataset):
    def __init__(self, input_ids, attention_masks, content_labels, wording_labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.content_labels = content_labels
        self.wording_labels = wording_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'content_label': self.content_labels[idx],
            'wording_label': self.wording_labels[idx]
        }

dataset = SummarizationDataset(input_ids, attention_masks, summaries_train['content'].values, summaries_train['wording'].values)

# Splitting data
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Model training preparations
# Model training preparations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.MSELoss()

# Training loop for two separate models
NUM_EPOCHS = 3
for score_type in ["content", "wording"]:
    print(f"Training model for {score_type} score...")
    
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch[f'{score_type}_label'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1), labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})
    
    model_save_path = f"/kaggle/working/bert_model_{score_type}"
    model.save_pretrained(model_save_path)


# Model inference
summaries_test = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv')
summaries_test['lemmatized_text'] = summaries_test['text'].apply(lemmatize_text)
input_ids_test, attention_masks_test = tokenize_for_bert(summaries_test['lemmatized_text'], MAX_LEN)
test_dataset = TensorDataset(input_ids_test, attention_masks_test)
test_dataloader = DataLoader(test_dataset, batch_size=16)

predictions = {"content": [], "wording": []}

for score_type in ["content", "wording"]:
    print(f"Predicting {score_type} score...")
    model_save_path = f"/kaggle/working/bert_model_{score_type}"
    model = BertForSequenceClassification.from_pretrained(model_save_path, num_labels=1).to(device)
    model.eval()
    for batch in test_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            predictions[score_type].extend(logits.squeeze().tolist())

# Create submission dataframe
submission_df = pd.DataFrame({
    'student_id': summaries_test['student_id'],
    'content': predictions["content"],
    'wording': predictions["wording"]
})
submission_df.to_csv('submission.csv', index=False)




print(submission_df.head())
print("Number of rows in submission:", len(submission_df))
print(submission_df.isnull().sum())
print(submission_df.dtypes)

