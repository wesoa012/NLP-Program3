# REPORT: Guide on fine-tuning `BERT` on word disambiguation

**NOTE**: The following section assume you're running on SMU **SuperPod**

## The Task

(Wes or Kassi or both)

Let's use this section to describe the task, the problem we are set to solve, and in one sentence or two, state our approach, which consisted in using BERT. But stated nicely.

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

(Wes)
Let's use this section to describe the three datasets we did use, let's give the specs, also let's describe How we augmented it, how we curated it, and add any other relevant information that pertains to the dataset

## How did we train the models?

(Kassi or Wes)

## How to perform inference?

(Kassi)

First enter the `WSD` folder, inside the `bert-fine-tuning` folder using:

```sh
cd bert-fine-tuning/WSD/
```

## Performance metrics

(Kassi)
I will put a table here, with performance on testing sets. On different runs. It's just to flex 💪