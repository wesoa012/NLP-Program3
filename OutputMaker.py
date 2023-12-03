import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import DistilBertTokenizerFast
from bert_fine_tuning.WSD.WSDDataset import WSDDataset
from bert_fine_tuning.WSD.wsd_classifier import WSDClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def WSD_Test_Rubbish(sentences):
    output = open("./rubbish_outputs.txt", "w")
    
    encodings = TOKENIZER(sentences, truncation=True, padding=True)
    
    for encoding in encodings:
        output.write(str(encoding + 1) + "\n")
        
def WSD_Test_Thorn(sentences):
    output = open("./thorn_outputs.txt", "w")
    
    encodings = TOKENIZER(sentences, truncation=True, padding=True)
    
    for encoding in encodings:
        output.write(str(encoding + 1) + "\n")
        
def WSD_Test_Conviction(sentences):
    output = open("./conviction_outputs.txt", "w")
    
    encodings = TOKENIZER(sentences, truncation=True, padding=True)
    
    for encoding in encodings:
        output.write(str(encoding + 1) + "\n")
