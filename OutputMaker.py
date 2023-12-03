import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import DistilBertTokenizerFast
from bert_fine_tuning.WSD.wsd_classifier import WSDClassifier

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def WSD_Test_Rubbish(sentences):
    # Load the model and its weights on the GPU
    rubbish_BERT = WSDClassifier()
    rubbish_BERT.to(DEVICE)
    rubbish_BERT.load_model('bert_fine_tuning/WSD/models/rubbish_network_weights.pth')
    
    # Send the sentences to the tokenizer, and get the encodings
    encodings = TOKENIZER(sentences, truncation=True, padding=True)
    
    # Perform inference by sending the encodings to the Rubbish BERT
    with torch.no_grad():
        input_ids = torch.tensor(encodings['input_ids']).to(DEVICE)
        attention_mask = torch.tensor(encodings['attention_mask']).to(DEVICE)
        outputs = rubbish_BERT(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1) # "predicted_labels" is a torch tensor on the GPU.

    predicted_labels_cpu = predicted_labels.cpu() # Move tensor from GPU to CPU
    predicted_labels_list = predicted_labels_cpu.tolist() # Convert tensor to a Python list of integers

    # Save the results to the file.
    output = open("./result_rubbish_WesAnderson.txt", "w") # IMPORTANT:  “result_<word>_<first and last name of one member of your group>.txt” format
    for i, label in enumerate(predicted_labels_list):
        if i == len(predicted_labels_list) - 1:
            output.write(str(label + 1))
        else:
            output.write(str(label + 1) + "\n")
        
def WSD_Test_Thorn(sentences):
    # Load the model and its weights on the GPU
    thorn_BERT = WSDClassifier()
    thorn_BERT.to(DEVICE)
    thorn_BERT.load_model('bert_fine_tuning/WSD/models/thorn_network_weights.pth')

    encodings = TOKENIZER(sentences, truncation=True, padding=True)

    # Perform inference by sending the encoding to the Thorn BERT
    with torch.no_grad():
        input_ids = torch.tensor(encodings['input_ids']).to(DEVICE)
        attention_mask = torch.tensor(encodings['attention_mask']).to(DEVICE)
        outputs = thorn_BERT(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
    
    predicted_labels_cpu = predicted_labels.cpu()
    predicted_labels_list = predicted_labels_cpu.tolist()
    
    # Save the results to the file
    output = open("./result_thorn_WesAnderson.txt", "w") # IMPORTANT:  “result_<word>_<first and last name of one member of your group>.txt” format
    for i, label in enumerate(predicted_labels_list):
        if i == len(predicted_labels_list) - 1:
            output.write(str(label + 1))
        else:
            output.write(str(label + 1) + "\n")
        
def WSD_Test_Conviction(sentences):
    # Load the model and its weights on the GPU
    conviction_BERT = WSDClassifier()
    conviction_BERT.to(DEVICE)
    conviction_BERT.load_model('bert_fine_tuning/WSD/models/conviction_network_weights.pth')

    encodings = TOKENIZER(sentences, truncation=True, padding=True)

    # Perform inference by sending the encodings to the Conviction BERT
    with torch.no_grad():
        input_ids = torch.tensor(encodings['input_ids']).to(DEVICE)
        attention_mask = torch.tensor(encodings['attention_mask']).to(DEVICE)
        outputs = conviction_BERT(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

    predicted_labels_cpu = predicted_labels.cpu()
    predicted_labels_list = predicted_labels_cpu.tolist()

    # Save the results to the file
    output = open("./result_conviction_WesAnderson.txt", "w") # IMPORTANT:  “result_<word>_<first and last name of one member of your group>.txt” format
    for i, label in enumerate(predicted_labels_list):
        if i == len(predicted_labels_list) - 1:
            output.write(str(label + 1))
        else:
            output.write(str(label + 1) + "\n")


if __name__ == "__main__":
    rubbish_sentences = []
    thorn_sentences = []
    conviction_sentences = []

    # Read Dr.Lin's files with sentences to be disambiguated, into separate Python lists
    try:
        with open('./rubbish_test.txt', 'r') as file:
            # Read all lines into a list
            rubbish_sentences = file.readlines()
            rubbish_sentences = [sentence.strip() for sentence in rubbish_sentences]
    except FileNotFoundError:
        print(f"File './rubbish_test.txt' not found.")
        sys.exit(0)

    try:
        with open('./thorn_test.txt', 'r') as file:
            # Read all lines into a list
            thorn_sentences = file.readlines()
            thorn_sentences = [sentence.strip() for sentence in thorn_sentences]
    except FileNotFoundError:
        print(f"File './thorn_test.txt' not found.")
        sys.exit(0)

    try:
        with open('./conviction_test.txt', 'r') as file:
            # Read all lines into a list
            conviction_sentences = file.readlines()
            conviction_sentences = [sentence.strip() for sentence in conviction_sentences]
    except FileNotFoundError:
        print(f"File './conviction_test.txt' not found.")
        sys.exit(0) 
    
    # Send each group of sentences to right BERT model.
    WSD_Test_Rubbish(rubbish_sentences)
    WSD_Test_Thorn(thorn_sentences)
    WSD_Test_Conviction(conviction_sentences)