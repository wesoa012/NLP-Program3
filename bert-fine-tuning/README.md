# Guide on how to finetune BERT on Sentiment Analysis

### Acquire a Pytorch image

In `${HOME}` directory, create a `pytorch` directory using:

```sh
mkdir pytorch
```

Now, using the following command, download the Pytorch docker image:

```sh
enroot import -o ${HOME}/pytorch/pytorch.sqsh docker://pytorch/pytorch:latest
```

**NOTE: The image will be downloaded and stored in the `pytorch` folder you created earlier. Fill free to store it anywhere else. Just make sure to update the path accordingly**

### Acquire an interactive compute node on SuperPod

Using the following command, will acquire a compute node with one GPU, and 4 CPU. The container will be interactive so you can type commands.

```sh
srun -N1 -G1 -c4 --mem=100G --container-remap-root --no-container-entrypoint --container-image ${HOME}/pytorch/pytorch.sqsh --container-mounts="${HOME}"/bert-fine-tuning:/workdir --container-workdir /workdir --pty bash -i
```

### Now setup the environment.

Assuming you acquired a compute node on superpod running pytorch images, you can setup the environment using the following the `env-setup.sh` script using:

```sh
./env-setup.sh
```
It will download all the Python packages needed to run the code.That's it. 

### Perform inference on the model

If everything went according to plan, head into the sentiment folder:

```sh
cd sentiment
```

Now, try to run the `BERT` sentiment classifier using:

```sh
python main.py
```

**NOTE: This command will either load the model weights if the model has been train before OR start training the model.**

# Okay great, but we are doing word disambiguation, right?

# Setting up the environment

If you ran the above command, you already ready... No new package to install 🚀

Yes, with all we need to do is: Swap the dataset. Simple as that.

Okay well... not *that* simple. Let me walk through how to swap the dataset.

The first thing different is that we won't be fine-tuning one model, but three. Why? For the program 3, we have three words:

- conviction
- rubbish
- thorn



We are going to train **three** BERT, **one for each word**. Each word will trained to classify one the sense of the 
