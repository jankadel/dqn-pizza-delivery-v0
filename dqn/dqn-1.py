import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pygame
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

ENV = gym.make("gym_pizza_delivery:gym_pizza_delivery-v0")

IS_IPYTHON = 'inline' in matplotlib.get_backend()
if IS_IPYTHON:
    from IPython import display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSITION = namedtuple('TRANSITION', ('state', 'action', 'follow_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def store(self, *args, keep=None):
        if keep and len(self.memory) != 0:
            head = self.memory.pop()
            self.memory.append(head)
        else:
            self.memory.append(TRANSITION(*args))

    def sample(self, range):
        return random.sample(self.memory, range)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, height, width, outputs) -> None:
        super(DQN, self).__init__()
        # (6,4;3,1;2,1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=6, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        #print(height, width)
        def get_conv2d_size(size, kernel_size=5, stride=2):
            return (size-(kernel_size-1)-1) // stride+1

        conv_w = get_conv2d_size(get_conv2d_size(get_conv2d_size(width)))
        conv_h = get_conv2d_size(get_conv2d_size(get_conv2d_size(height)))
        
        lin_input_size = conv_h * conv_w * 32

        self.head = nn.Linear(lin_input_size, outputs)

    def forward(self, x):
        x = x.to(DEVICE)
        x = F.relu(self.bn1(self.pool(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(100, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(img=None):
    if not img:
        screen = ENV.render() # type(screen) = pygame.Surface
        width = screen.get_width()
        height = screen.get_height()
        surface = pygame.image.tostring(screen, 'RGB')
        img = Image.frombytes('RGB', (width, height), surface) # raw byte string
    img_arr = np.array(img)
    img_arr = img_arr.transpose((2, 0, 1)) # Transpose to torch order (CHW)
    img_arr = np.ascontiguousarray(img_arr, dtype=np.float32)
    screen = torch.from_numpy(img_arr)
    return resize(screen).unsqueeze(0)

#### CNN
NETWORK = "DQN"

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.999
EPSILON_START = 0.95
EPSILON_END = 0.05
DECAY = 200
TARGET_UPDATE = 5
MEMSIZE = 10000
MEMORY = ReplayMemory(MEMSIZE)
STEPS = 0
DURATIONS = []

init_screen = get_screen()
screen_height = init_screen.shape[2]
screen_width = init_screen.shape[3]

num_actions = ENV.action_space.n
observation_space = ENV.observation_space
  
Q_net = DQN(screen_height, screen_width, num_actions).to(DEVICE)
target_net = DQN(screen_height, screen_width, num_actions).to(DEVICE)
target_net.load_state_dict(Q_net.state_dict())
target_net.eval()

OPTIMIZER = optim.RMSprop(Q_net.parameters(), lr=LEARNING_RATE)

def get_action(state):
    global STEPS
    STEPS += 1
    sample = random.random()
    threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1 * STEPS / DECAY)
    if sample > threshold:
        with torch.no_grad():
            return Q_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(num_actions)]], device=DEVICE, dtype=torch.long)

def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(DURATIONS, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if IS_IPYTHON:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize():
    if len(MEMORY) < BATCH_SIZE:
        return
    transitions = MEMORY.sample(BATCH_SIZE)
    batch = TRANSITION(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.follow_state)), device=DEVICE, dtype=torch.bool)
    non_final_follow_states = torch.cat([s for s in batch.follow_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values = Q_net(state_batch).gather(1, action_batch)
    follow_states = torch.zeros(BATCH_SIZE, device=DEVICE)
    follow_states[non_final_mask] = target_net(non_final_follow_states).max(1)[0].detach()
    e_q_values = (follow_states * GAMMA) + reward_batch

    # Huber Loss
    huber = nn.SmoothL1Loss()
    loss = huber(q_values, e_q_values.unsqueeze(1))
    if torch.cuda.is_available():
        loss_val = copy.deepcopy(loss.cpu().detach().numpy())
    else:
        loss_val = copy.deepcopy(loss.detach().numpy())
    
    # optimize model
    OPTIMIZER.zero_grad()
    loss.backward()
    for p in Q_net.parameters():
        p.grad.data.clamp_(-1, 1)
    OPTIMIZER.step()
    return loss_val

def train(num_episodes : int = 100):
    plt.ion()
    eps_rewards = []
    sum_rewards = 0
    eps_losses = []
    losses = []
    for e in range(1, num_episodes):
        start = time.time()
        print("Episode:", e)
        ENV.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            action = get_action(state)
            obs, reward, done, _ = ENV.step(action.item())
            reward = torch.tensor([reward], device=DEVICE)
            #print(reward)
            last_screen = copy.deepcopy(current_screen)
            current_screen = get_screen(obs)

            if done:
                follow_state = None
            else:
                follow_state = current_screen - last_screen

            if t % 500 == 0:
                MEMORY.store(state, action, follow_state, reward, keep=True)
            else:
                MEMORY.store(state, action, follow_state, reward)
            state = copy.deepcopy(follow_state)
            
            if t % 100000 == 0:
                print(t)

            loss = optimize()

            # Collect data for episode
            losses.append(loss)
            sum_rewards += reward
            if done:
                DURATIONS.append(t + 1)
                # Store episode data (episode, steps, cumulative reward, avg loss)
                eps_rewards.append([e, t, sum_rewards])
                eps_losses.append([e, losses])
                sum_rewards = 0
                losses = []
                plot_durations()
                end = time.time()
                print(f"Episode {e} finished after {t} iterations or {end - start} seconds.")
                break

            if e % TARGET_UPDATE == 0:
                target_net.load_state_dict(Q_net.state_dict())
            ENV.render()
            if sum_rewards > 0:
                torch.save(Q_net.state_dict(), f"./models/q_net_eps_{e}.txt")
                torch.save(target_net.state_dict(), f"./models/target_net_eps_{e}.txt")

    print('Training Completed')
    with open("./data/eps_rewards.txt", 'w') as f:
        for item in eps_rewards:
            f.write("%s\n" % item)
    with open("./data/eps_losses.txt", 'w') as f:
        for item in eps_losses:
            f.write("%s\n" % item)
    ENV.close()
    plt.ioff()
    plt.show()

def predict(state, model):
    with torch.no_grad():
        return model(state).max(1)[1].view(1, 1)

def infer(episodes:int, model_path:str):
    model = DQN(screen_height, screen_width, num_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    rewards = []
    sum_reward = 0

    current_screen = get_screen()
    last_screen = copy.deepcopy(init_screen)
    state = current_screen - last_screen
    for i in range(episodes):
        current_screen = get_screen()
        state = current_screen - last_screen
        action = get_action(state)
        obs, reward, done, _ = ENV.step(action.item())

        sum_reward += reward
        rewards.append(sum_reward)

        last_screen = copy.deepcopy(current_screen)
        current_screen = get_screen(obs)
        
        if done:
            ENV.reset()

    ENV.close()
    plt.figure(1)
    plt.title('Inference Mode - DQL')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.plot(rewards)

    plt.show()


#train(num_episodes=1000)
infer(episodes=10000, model_path="./models/exp2_models/q_net_eps_7.txt")
