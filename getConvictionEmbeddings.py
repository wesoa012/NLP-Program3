import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from transformers import DistilBertTokenizerFast
from bert_fine_tuning.WSD.wsd_classifier import WSDClassifier
from sklearn.decomposition import PCA


# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

sense1_sentences,sense2_sentences = [],[]

with open('./Data/txts/Conviction.txt', 'r') as file:
    for line in file.readlines():
        if 'conviction' not in line:
            continue
        else:
            if line[0] == '1':
                sense1_sentences.append(line[2:])
            else:
                sense2_sentences.append(line[2:])

print(f"lengths = {len(sense1_sentences)}, {len(sense1_sentences)}")

#get hidden vector for each "thorn" token
    #plot the vectors using PCA
#get cosine similarity between all pairs.
    #plot values of cosine similarities

sense1_encodings = TOKENIZER(sense1_sentences, truncation=True, padding=True)
sense2_encodings = TOKENIZER(sense2_sentences, truncation=True, padding=True)

conviction_BERT = WSDClassifier()
conviction_BERT.to(DEVICE)
conviction_BERT.load_model('bert_fine_tuning/WSD/models/conviction_network_weights.pth')
conviction_BERT.eval()


with torch.no_grad():

    sense1_tokens_tensor = torch.tensor(sense1_encodings['input_ids']).to(DEVICE)
    sense1_segments_tensors = torch.tensor(sense1_encodings['attention_mask']).to(DEVICE)
    sense1_outputs = conviction_BERT(sense1_tokens_tensor, sense1_segments_tensors)
    sense1_hidden_states = sense1_outputs.hidden_states

    sense2_tokens_tensor = torch.tensor(sense2_encodings['input_ids']).to(DEVICE)
    sense2_segments_tensors = torch.tensor(sense2_encodings['attention_mask']).to(DEVICE)
    sense2_outputs = conviction_BERT(sense2_tokens_tensor, sense2_segments_tensors)
    sense2_hidden_states = sense2_outputs.hidden_states


sense1_what_we_care_abouts = []
for i in range(len(sense1_hidden_states[0])):
    for j in range(len(sense1_encodings[i])):
        if sense1_encodings[i].tokens[j] == 'conviction':
            sense1_what_we_care_abouts.append(sense1_hidden_states[0][i][j]) 
            break

transformed_sense1_x, transformed_sense1_y = [],[]
for i in sense1_what_we_care_abouts:
    to_transform = torch.Tensor.cpu(i)
    to_transform = np.array(to_transform)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(to_transform.reshape(-1,2))
    transformed_sense1_x.append(transformed[:, 0])
    transformed_sense1_y.append(transformed[:, 1])

sense2_what_we_care_abouts = []
for i in range(len(sense2_hidden_states[0])):
    for j in range(len(sense2_encodings[i])):
        if sense2_encodings[i].tokens[j] == 'conviction':
            sense2_what_we_care_abouts.append(sense2_hidden_states[0][i][j]) 
            break

transformed_sense2_x, transformed_sense2_y = [],[]
for i in sense2_what_we_care_abouts:
    to_transform = torch.Tensor.cpu(i)
    to_transform = np.array(to_transform)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(to_transform.reshape(-1,2))
    # print()
    transformed_sense2_x.append(transformed[:, 0])
    transformed_sense2_y.append(transformed[:, 1])


fig, ax = plt.subplots(figsize=(12,7))

ax.set_title("Plotting Sentence Embeddings after run through BERT",fontsize=20)

plt.scatter(transformed_sense1_x, transformed_sense1_y, c='b', alpha=0.5, label="Sense 1")
plt.scatter(transformed_sense2_x, transformed_sense2_y, c='r', alpha=0.5, label="Sense 2")

print(f"num sense1s = {len(transformed_sense1_x)}")
print(f"num sense2s = {len(transformed_sense2_x)}")


plt.legend(loc = 'best')

plt.savefig('./Graphs/Sentence_Embeddings_Conviction.png')

transformed_sense1_x, transformed_sense1_y = [],[]
for i in sense1_what_we_care_abouts:
    to_transform = torch.Tensor.cpu(i)
    to_transform = np.array(to_transform)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(to_transform.reshape(2,-1))
    transformed_sense1_x.append(transformed[:, 0])
    transformed_sense1_y.append(transformed[:, 1])

transformed_sense2_x, transformed_sense2_y = [],[]
for i in sense2_what_we_care_abouts:
    to_transform = torch.Tensor.cpu(i)
    to_transform = np.array(to_transform)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(to_transform.reshape(2,-1))
    transformed_sense2_x.append(transformed[:, 0])
    transformed_sense2_y.append(transformed[:, 1])


fig, ax = plt.subplots(figsize=(12,7))

ax.set_title("Plotting Sentence Embeddings after run through BERT",fontsize=20)

plt.scatter(transformed_sense1_x, transformed_sense1_y, c='b', alpha=0.5, label="Sense 1")
plt.scatter(transformed_sense2_x, transformed_sense2_y, c='r', alpha=0.5, label="Sense 2")

print(f"num sense1s = {len(transformed_sense1_x)}")
print(f"num sense2s = {len(transformed_sense2_x)}")


plt.legend(loc = 'best')

plt.savefig('./Graphs/Sentence_Embeddings_Flipped_Conviction.png')


cos_sims_sense1_only, cos_sims_sense2_only, cos_sims_both = [],[],[]
do_sense1_only, do_sense2_only, do_both_senses = True,True,True
if(os.path.exists('./Data/embeddings/conviction_cos_sims_sense1_only.csv')):
    print("skipping creation of cos_sims for sense 1")
    do_sense1_only = False
if(os.path.exists('./Data/embeddings/conviction_cos_sims_sense2_only.csv')):
    print("skipping creation of cos_sims for sense 2")
    do_sense2_only = False
if(os.path.exists('./Data/embeddings/conviction_cos_sims_both_senses.csv')):
    print("skipping creation of cos_sims for both senses")
    do_both_senses = False

if(do_sense1_only or do_both_senses):
    for i in range(len(sense1_what_we_care_abouts)):
        vec_sense1 = sense1_what_we_care_abouts[i]
        a = torch.Tensor.cpu(vec_sense1)
        if(do_sense1_only and len(cos_sims_sense1_only) < 1000000):
            for j in range(len(sense1_what_we_care_abouts)):
                vec_sense2 = sense1_what_we_care_abouts[j]
                b = torch.Tensor.cpu(vec_sense2)
                if i == j:
                    continue
                bottom = np.linalg.norm(a)*np.linalg.norm(b)
                if bottom == 0:
                    cos_sims_sense1_only.append(0)
                else:
                    cos_sims_sense1_only.append(np.dot(a,b)/bottom)

        if(do_both_senses and len(cos_sims_both) < 1000000):
            for j in range(len(sense2_what_we_care_abouts)):
                vec_sense2 = sense2_what_we_care_abouts[j]
                b = torch.Tensor.cpu(vec_sense2)
                bottom = np.linalg.norm(a)*np.linalg.norm(b)
                if bottom == 0:
                    cos_sims_both.append(0)
                else:
                    cos_sims_both.append(np.dot(a,b)/bottom)

if(do_sense2_only):
    for i in range(len(sense2_what_we_care_abouts)):
        vec_sense1 = sense2_what_we_care_abouts[i]
        a = torch.Tensor.cpu(vec_sense1)
        if(len(cos_sims_sense2_only) < 1000000):
            for j in range(len(sense2_what_we_care_abouts)):
                vec_sense2 = sense2_what_we_care_abouts[j]
                b = torch.Tensor.cpu(vec_sense2)
                if i == j:
                    continue
                bottom = np.linalg.norm(a)*np.linalg.norm(b)
                if bottom == 0:
                    cos_sims_sense2_only.append(0)
                else:
                    cos_sims_sense2_only.append(np.dot(a,b)/bottom)


columns = ['Similarity Values']
if(do_sense1_only):
    print("Making dataframe/csv for cosine similarities for sense 1 only")
    cos_sims_sense1_only_df = pd.DataFrame(cos_sims_sense1_only)
    cos_sims_sense1_only_df.columns = columns
    cos_sims_sense1_only_df.to_csv('./Data/embeddings/conviction_cos_sims_sense1_only.csv')
else:
    print("Reading dataframe/csv for cosine similarities for sense 1 only")
    cos_sims_sense1_only_df = pd.read_csv('./Data/embeddings/conviction_cos_sims_sense1_only.csv')

if(do_sense2_only):
    print("Making dataframe/csv for cosine similarities for sense 2 only")
    cos_sims_sense2_only_df = pd.DataFrame(cos_sims_sense2_only)
    cos_sims_sense2_only_df.columns = columns
    cos_sims_sense2_only_df.to_csv('./Data/embeddings/conviction_cos_sims_sense2_only.csv')
else:
    print("Reading dataframe/csv for cosine similarities for sense 2 only")
    cos_sims_sense2_only_df = pd.read_csv('./Data/embeddings/conviction_cos_sims_sense2_only.csv')

if(do_both_senses):
    print("Making dataframe/csv for cosine similarities for both senses")
    cos_sims_both_df = pd.DataFrame(cos_sims_both)
    cos_sims_both_df.columns = columns
    cos_sims_both_df.to_csv('./Data/embeddings/conviction_cos_sims_both_senses.csv')
else:
    print("Reading dataframe/csv for cosine similarities for both senses")
    cos_sims_both_df = pd.read_csv('./Data/embeddings/conviction_cos_sims_both_senses.csv')


fig, ax = plt.subplots(figsize=(12,7))
cos_sims_sense1_only_df['Similarity Values'].plot(kind='kde')
ax.set_title(f'Sense 1 Only Similarity Distributions',fontsize=20)
ax.set_xlabel('Similarity Value',fontsize=15)
ax.set_ylabel('Density',fontsize=15)
plt.savefig('./Graphs/Conviction_Sense1_Only_SimVals.png')

fig, ax = plt.subplots(figsize=(12,7))
cos_sims_sense2_only_df['Similarity Values'].plot(kind='kde')
ax.set_title(f'Sense 2 Only Similarity Distributions',fontsize=20)
ax.set_xlabel('Similarity Value',fontsize=15)
ax.set_ylabel('Density',fontsize=15)
plt.savefig('./Graphs/Conviction_Sense2_Only_SimVals.png')

fig, ax = plt.subplots(figsize=(12,7))
cos_sims_both_df['Similarity Values'].plot(kind='kde')
ax.set_title(f'Both Senses Similarity Distributions',fontsize=20)
ax.set_xlabel('Similarity Value',fontsize=15)
ax.set_ylabel('Density',fontsize=15)
plt.savefig('./Graphs/Conviction_Both_Senses_SimVals.png')