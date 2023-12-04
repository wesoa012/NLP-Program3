# REPORT: Guide on fine-tuning `BERT` on word disambiguation

**NOTE**: The following section assume you're running on SMU **SuperPod**

## The Task

The problem we tackled in this program is to classify a word in different contexts as either sense 1 or sense 2. We built a separate classifier for each of the words: thorn, rubbish, and conviction. We describe in more detail below exactly how we trained these classifiers as well as their performance.

## Setting up the environment

### Acquire a Pytorch image

In `${HOME}` directory on SuperPod, create a `pytorch` directory using:

```sh
mkdir pytorch
```

Now, using the following command, download the Pytorch docker image:

```sh
enroot import -o ${HOME}/pytorch/pytorch.sqsh docker://pytorch/pytorch:latest
```

**NOTE: The image will be downloaded and stored in the `pytorch` folder you created earlier. Fill free to store it anywhere else. Just make sure to update the path accordingly**

### Clone the repository

In your `${HOME}` directory on SuperPod, clone this repository using:

```sh
git clone https://github.com/wesoa012/NLP-Program3.git
```

and enter it using:

```sh
cd NLP-Program3
```

### Acquire an interactive compute node on SuperPod

Now that you're at the root of the repository, use the following command to acquire a compute node with one GPU, and 4 CPU on SuperPod. The container will be interactive so you can type commands.

```sh
srun -N1 -G1 -c4 --mem=100G --container-remap-root --no-container-entrypoint --container-image ${HOME}/pytorch/pytorch.sqsh --container-mounts="${HOME}"/NLP-Program3/bert-fine-tuning:/workdir --container-workdir /workdir --pty bash -i
```

This command mounts the **`"${HOME}"/NLP-Program3/bert-fine-tuning`** folder on SuperPod to the **`/workdir`** folder in the container, and the `--container-workdir` flag on the command sets your working directory to be `workdir`.

*NOTE: If the `srun` command returns an error, once again check your paths, and make sure everything is installed at the expected location.*

*NOTE: Since SuperPod is allocated based on resource availability, you may or may not have a hard time acquiring a compute node. It will depend on how busy the system is when you try.*

### Install the dependencies

Inside your shiny new compute node, use the following command to install all the dependencies needed to run the program:

```sh
./env-setup.sh
```


## The Dataset

Since BERT works better with larger datasets we beefed up our training data with synthetically generated sentences from ChatGPT. We wanted a large dataset that would still train in under 5 minutes on Superpod, so we chose to have 5000 newly generated sentences for each sense of the word in addition to the sentences already provided. For some of these sentences that ChatGPT generated it did not use the exact word "conviction" or "thorn" but instead used "convicted criminal" and "thorny path." At first, we were going to prune these data points from the dataset, but when we thought about it from the computer's perspective these examples should help increase the word sense that the computer has of "conviction" and "thorn." When we left these words in it did not affect the performance of the classifier negatively.

The put-together dataset we used is located in the "Data" folder under csvs because when reading in the data we took advantage of the pandas library to easily put csv files into dataframe objects. The other folders in the "Data" folder contain the disjointed dataset that the "DataReader.py" can put together.

## How did we train the models?

To train the model we took the 10000 sentences generated from ChatGPT as well as the ~50 sentences from the provided sentences and broke it into 3 chunks: training data, validation data, and testing data. The split was 80% for training, 10% for validation, and 10% for testing. After splitting the dataset into these three separate chunks we ran the training data through BERT to train and edit the weights of the underlying neural network. This network then verified it was being trained properly by predicting on the validation. This prediction is stored and output to determine the quality of the model. After 3 iterations of training and validating the classifier is finished and we send it the testing data that it has never seen before so that it can predict. After it has predicted on the test data we get an output to see exactly how accurate it was on data it has never seen before. Once that process is completed for each classifier then our training process is officially over.

## How to perform inference?

(Kassi)

First enter the `WSD` folder, inside the `bert-fine-tuning` folder using:

```sh
cd bert-fine-tuning/WSD/
```

## Performance metrics

(Kassi)
I will put a table here, with performance on testing sets. On different runs. It's just to flex 💪
