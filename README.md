# REPORT: Guide on fine-tuning `BERT` on word disambiguation

## The Task

The problem we tackled in this program is to classify a word in different contexts as either sense 1 or sense 2. We built a separate classifier for each of the words: thorn, rubbish, and conviction. We describe in more detail below exactly how we trained these classifiers as well as their performance.

## Setting up the environment

**NOTE**: The following section assume you're running on SMU **SuperPod**. So, log on SuperPod if not done already.

### Acquire a Pytorch image

In your `${HOME}` directory on SuperPod, create a `pytorch` directory using:

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

Now that you're at the root of the `NLP-Program3` repository, use the following command to acquire a compute node with one GPU, and 4 CPU on SuperPod. The container will be interactive so you can type commands.

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

We used the same neural network architecture, we just changed the dataset each time. So, for the conviction word, we use the architecture, train it on the conviction dataset, and **save** the weights. For the rubbish word, we re-initialized the neural network, and train it again on the rubbish dataset, then save the weights. We do the same thing for thorn word.

The benefit of this approach was that we did not have to write three different neural networks. We wrote one. And when it's time to disambiguate a word, we simply load weights that correspond to the words and we are ready.

if you enter the `WSD` folder, inside the `bert-fine-tuning` folder using:

```sh
cd bert-fine-tuning/WSD/
```

Inside the `bert-fine-tuning/WSD/` folder, a quick `ls` command reveals the following files:

```txt
-rw-rw-r-- 1 knzalasse knzalasse 3458 Dec  2 18:15 conviction-net.py
drwxrwxr-x 2 knzalasse knzalasse 4096 Dec  2 18:59 models
drwxrwxr-x 2 knzalasse knzalasse 4096 Dec  2 18:10 __pycache__
-rw-rw-r-- 1 knzalasse knzalasse 3433 Dec  2 18:54 rubbish-net.py
-rw-rw-r-- 1 knzalasse knzalasse 3439 Dec  2 18:57 thorn-net.py
-rw-rw-r-- 1 knzalasse knzalasse 3257 Nov 30 16:46 wsd_classifier.py
-rw-rw-r-- 1 knzalasse knzalasse  417 Nov 24 18:44 WSDDataset.py
```

The `models` folder contains the weights of the three models. But if it's your first time going through this procedure or you did not train the models yet, the `models` folder might be non-existant.

The following files contain the source code for the three BERT we trained. One file for each word

1. `conviction-net.py`
2. `rubbish-net.py`
2. `thorn-net.py`

To train train the BERT for the `conviction` word, run

```sh
python conviction-net.py
```

To train the BERT for the `rubbish` word, run

```sh
python rubbish-net.py
```

To train the BERT for the `thorn` word, run:

```sh
python thorn-net
```

Each of these command should take about 5min to train the network.

## How to perform inference?

Make sure you trained the models, before trying to perform inference. If you've trained the models, the following command:

```sh
ls -l models/
```

will return the following:

```txt
-rw-rw-r-- 1 knzalasse knzalasse 267863654 Dec  2 18:28 conviction_network_weights.pth
-rw-rw-r-- 1 knzalasse knzalasse 267863330 Dec  2 18:45 rubbish_network_weights.pth
-rw-rw-r-- 1 knzalasse knzalasse 267863050 Dec  2 18:59 thorn_network_weights.pth
```

Now that we made sure that the weights are present, let's go back to the root of the repo, which is the `NLP-Program3` folder using:

```sh
cd ../../
```
You can make sure you're in the right folder using the `pwd` command.

To perform inference, run the `cs5322f23.py` script using:

```sh
python cs5322f23.py
```

## Performance Benchmark

As previously mentioned, the `test` split allowed us to test the performance of the model on unseen data. Here are the performance of the model during testing. The model was ran in evaluation mode.

| Model          | Performance |
|----------------|-------------|
| conviction-net | 95%         |
| rubbish-net    | 98%         |
| thorn-net      | 98%         |

**NOTE:** The performance might differ on your machine since you're training your own network from sratch.
