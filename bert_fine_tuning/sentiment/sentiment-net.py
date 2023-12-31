import os
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import DistilBertTokenizerFast
from IMDbDataset import IMDbDataset
from sentiment_classifier import SentimentClassifier

# constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

if __name__ == '__main__':
    model_path = 'models/sentiment_classifier.pth'

    # Check if the 'models' folder exists, if not, create it
    if not os.path.exists('models'):
        os.makedirs('models')

    # Check if the model weights are saved, if yes load them... otherwise train the model.
    if os.path.exists(model_path):
        print("Found saved model. Loading...")
        model = SentimentClassifier()
        model.to(DEVICE)
        model.load_model(model_path)
        model.eval()
    else:
        print("No saved model found. Training...")
        
        # Read in the dataset in a Pandas dataframe. Total: 50,000 = 25,000(0) + 25,000(1)
        df = pd.read_csv('movie_data.csv')

        # Split the dataset into Train/Valid/Test sets
        train_texts = df.iloc[:35000]['review'].values #35000 sentences
        train_labels = df.iloc[:35000]['sentiment'].values

        valid_texts = df.iloc[35000:40000]['review'].values # 5000 sentences
        valid_labels = df.iloc[35000:40000]['sentiment'].values

        test_texts = df.iloc[40000:]['review'].values # 10,000 sentences
        test_labels = df.iloc[40000:]['sentiment'].values

        # Tokenizing each split of the dataset
        train_encodings = TOKENIZER(list(train_texts), truncation=True, padding=True)
        valid_encodings = TOKENIZER(list(valid_texts), truncation=True, padding=True)
        test_encodings = TOKENIZER(list(test_texts), truncation=True, padding=True)

        # Create torch datasets out of the splits
        train_dataset = IMDbDataset(train_encodings, train_labels)
        valid_dataset = IMDbDataset(valid_encodings, valid_labels)
        test_dataset = IMDbDataset(test_encodings, test_labels)

        # Create the dataloaders. 
        # Dataloaders create iteratable objects out of the dataset objects so we can iterate on the datasets using for loops.
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Fine tuning the model
        model = SentimentClassifier()
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