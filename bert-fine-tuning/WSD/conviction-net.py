import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizerFast
from WSDDataset import WSDDataset
from wsd_classifier import WSDClassifier # The BERT neural network

# constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

if __name__ == '__main__':
    model_path = 'models/conviction_network_weights.pth'

    # Check if the 'models' folder exists, if not, create it
    if not os.path.exists('models'):
        os.makedirs('models')

    # Check if the model weights are saved, if yes load them... otherwise train the model.
    if os.path.exists(model_path):
        print("Found saved model. Loading...")
        model = WSDClassifier()
        model.to(DEVICE)
        model.load_model(model_path)
        model.eval()
    else:
        print("No saved model found. Training...")

        df = pd.read_csv('/workdir/Data/csvs/Conviction.csv', header=None, names=['sense','sentence'])

        # Split the dataset into Train/Valid/Test sets
        train_sentences = df.iloc[:8000]['sentence'].values 
        train_labels = df.iloc[:8000]['sense'].values

        valid_sentences = df.iloc[8000:9000]['sentence'].values 
        valid_labels = df.iloc[8000:9000]['sense'].values

        test_sentences = df.iloc[9000:10000]['sentence'].values
        test_labels = df.iloc[9000:10000]['sense'].values

        # Tokenizing each split of the dataset
        train_encodings = TOKENIZER(list(train_sentences), truncation=True, padding=True)
        valid_encodings = TOKENIZER(list(valid_sentences), truncation=True, padding=True)
        test_encodings = TOKENIZER(list(test_sentences), truncation=True, padding=True)

        # Create torch datasets out of the splits
        train_dataset = WSDDataset(train_encodings, train_labels)
        valid_dataset = WSDDataset(valid_encodings, valid_labels)
        test_dataset = WSDDataset(test_encodings, test_labels)

        # Create the dataloaders. 
        # Dataloaders create iteratable objects out of the dataset objects so we can iterate on the datasets using for loops.
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Fine tuning the model
        model = WSDClassifier()
        model.train_model(train_loader, valid_loader, test_loader, DEVICE)
        model.save_model(model_path)

    # With the model loaded, let's perform inference
    inference_texts = pd.read_csv('inference_data.txt', header=None, names=['sentences'])
    encodings = TOKENIZER(list(inference_texts['sentences']), truncation=True, padding=True)

    with torch.no_grad():
        input_ids = torch.tensor(encodings['input_ids']).to(DEVICE)
        attention_mask = torch.tensor(encodings['attention_mask']).to(DEVICE)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

    # View the predictions
    print("Predicted Labels:", predicted_labels)
    