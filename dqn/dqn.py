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
        # Convolutional Layer Parameters: (6,4;3,1;2,1) for 100px img, (5,4;3,2;2,2) for 40px img
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
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

class DQNagent:
    def __init__(self) -> None:
        self.resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

        self.batch_size = 128
        self.learning_rate = 0.001
        self.gamma = 0.999
        self.epsilon_start = 0.95
        self.epsilon_end = 0.05
        self.decay = 200
        self.target_update = 5
        self.memsize = 10000
        self.memory = ReplayMemory(self.memsize)
        self.steps = 0
        self.durations = []

        self.init_screen = self.get_screen()
        self.screen_height = self.init_screen.shape[2]
        self.screen_width = self.init_screen.shape[3]

        self.num_actions = ENV.action_space.n
        self.observation_space = ENV.observation_space
    
        self.Q_net = DQN(self.screen_height, self.screen_width, self.num_actions).to(DEVICE)
        self.target_net = DQN(self.screen_height, self.screen_width, self.num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.Q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.Q_net.parameters(), lr=self.learning_rate)
        


    def get_screen(self, img=None):
        if img is None:
            screen = ENV.render() # type(screen) = pygame.Surface
            width = screen.get_width()
            height = screen.get_height()
            surface = pygame.image.tostring(screen, 'RGB')
            img = Image.frombytes('RGB', (width, height), surface) # raw byte string
        img_arr = np.array(img)
        img_arr = img_arr.transpose((2, 0, 1)) # Transpose to torch order (CHW)
        img_arr = np.ascontiguousarray(img_arr, dtype=np.float32)
        screen = torch.from_numpy(img_arr)
        return self.resize(screen).unsqueeze(0)

    #### CNN


    def get_action(self, state):
        self.steps += 1
        sample = random.random()
        threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps / self.decay)
        if sample > threshold:
            with torch.no_grad():
                return self.Q_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=DEVICE, dtype=torch.long)

    def plot_durations(self):
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.durations, dtype=torch.float)
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

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = TRANSITION(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.follow_state)), device=DEVICE, dtype=torch.bool)
        non_final_follow_states = torch.cat([s for s in batch.follow_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values = self.Q_net(state_batch).gather(1, action_batch)
        follow_states = torch.zeros(self.batch_size, device=DEVICE)
        follow_states[non_final_mask] = self.target_net(non_final_follow_states).max(1)[0].detach()
        e_q_values = (follow_states * self.gamma) + reward_batch

        # Huber Loss
        huber = nn.SmoothL1Loss()
        loss = huber(q_values, e_q_values.unsqueeze(1))
        if torch.cuda.is_available():
            loss_val = copy.deepcopy(loss.cpu().detach().numpy())
        else:
            loss_val = copy.deepcopy(loss.detach().numpy())
        
        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.Q_net.parameters():
            p.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss_val

    def train(self, num_episodes : int = 100):
        plt.ion()
        eps_rewards = []
        sum_rewards = 0
        eps_losses = []
        losses = []
        for e in range(1, num_episodes):
            start = time.time()
            print("Episode:", e)
            ENV.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            for t in count():
                action = self.get_action(state)
                obs, reward, done, _ = ENV.step(action.item())
                reward = torch.tensor([reward], device=DEVICE)
                #print(reward)
                last_screen = copy.deepcopy(current_screen)
                current_screen = self.get_screen(obs)

                if done:
                    follow_state = None
                else:
                    follow_state = current_screen - last_screen

                if t % 500 == 0:
                    self.memory.store(state, action, follow_state, reward, keep=True)
                else:
                    self.memory.store(state, action, follow_state, reward)
                state = copy.deepcopy(follow_state)
                
                if t % 100000 == 0:
                    print(t)

                loss = self.optimize()

                # Collect data for episode
                losses.append(loss)
                sum_rewards += reward
                if done:
                    self.durations.append(t + 1)
                    # Store episode data (episode, steps, cumulative reward, avg loss)
                    eps_rewards.append([e, t, sum_rewards])
                    eps_losses.append([e, losses])
                    sum_rewards = 0
                    losses = []
                    self.plot_durations()
                    end = time.time()
                    print(f"Episode {e} finished after {t} iterations or {end - start} seconds.")
                    break

                if e % self.target_update == 0:
                    self.target_net.load_state_dict(self.Q_net.state_dict())
                ENV.render()
                if sum_rewards > 0:
                    torch.save(self.Q_net.state_dict(), f"./models/q_net_eps_{e}.txt")
                    torch.save(self.target_net.state_dict(), f"./models/target_net_eps_{e}.txt")

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

    def predict(self, state, model):
        with torch.no_grad():
            return model(state).max(1)[1].view(1, 1)

    def infer(self, episodes:int, model_path:str):
        model = DQN(self.screen_height, self.screen_width, self.num_actions)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(DEVICE)
        rewards = []
        sum_reward = 0

        current_screen = self.get_screen()
        last_screen = copy.deepcopy(self.init_screen)
        state = current_screen - last_screen
        for i in range(episodes):
            current_screen = self.get_screen()
            state = current_screen - last_screen
            action = self.predict(state, model)
            obs, reward, done, _ = ENV.step(action.item())

            sum_reward += reward
            rewards.append(sum_reward)

            last_screen = copy.deepcopy(current_screen)
            current_screen = self.get_screen(obs)
            
            if done:
                ENV.reset()

        ENV.close()
        plt.figure(1)
        plt.title('Inference Mode - DQL')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.plot(rewards)

        plt.show()