# DQN Pizza Delivery
GitHub repository for the implementation of Deep Q-Learning (DQN), which is a model-free reinforcement learning algorithm, used in this example to solve the delivery problem within a closed environment. The environment is given by a map with roads and non-road areas, in which the agent is able to move around freely. This means the agent can move up, down, left, right and deliver the item. In the environment `dqn-pizza-delivery-v0`, the item to be delivered is delicious virtual Pizza. 

DQN will take over the control of the agent and try to optimize the Pizza delivery process. It does that by using snapshots of the environment and then put those snapshots into a convolutional neural network(CNN), which processes the snapshots of the environment and outputs Q-values. On basis of those Q-values the agent takes an action.

The environment during training can be seen down below. Note that the agent is not performing very well on that gif, it was just provided to give you an impression of the environment.

![Pizza Delivery Environment](https://abload.de/img/env_anim9zjdc.gif)

## Setup 

To set everything up, you first must install the required dependencies. To do so, use the command:

`pip3 install -r requirements.txt`

Afterwards the environment can be installed via `pip3 install -e .`

For installation of PyTorch and CUDA, please first download NVIDIA CUDA (version must be at least 11.1) 

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#download-cuda-software

and then use the following command: 

`pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html` 

Now everything is set up and ready for use!

## Data

Data of pre-trained models can be downloaded here: https://drive.google.com/file/d/1U13u8G6LriU0cQdIRFWctRi165UAQ-WS/view?usp=sharing
The data could not be uploaded to GitHub because it is simply too large.

To use the data, just copy the *.txt files into the respective folders of the repository, i.e. models/model.txt(from zip archive) -> models/model.txt(repo) 

## Usage

To run the reinforcement learning algorithms, simply scroll down to the end of each file within the folder `dqn` and (un)comment the functions you want to run. Now add possible parameters, i.e. number of episodes or pre-trained model path. Then, if everything is set, save the file and run it with a Python interpreter.