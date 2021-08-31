# imports
from time import time
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt

class SB3DQNagent:
    """
    Class that uses a stable implementation of DQN by stable_baselines3 (see: https://stable-baselines3.readthedocs.io/en/master/) 
    This version of DQN is used as a benchmark for the own implementation of DQN
    """
    def __init__(self) -> None:
        """
        Function to initialize the SB3DQNagent and here only the pizza delivery environment
        """
        self.env = gym.make("gym_pizza_delivery:gym_pizza_delivery-v0")

    def train_sb3_dqn(self, timesteps):
        """
        Function that initializes the model and trains it.
        """
        model = DQN(
            "CnnPolicy",
            self.env,
            verbose=1,
            learning_starts=15000,
            gamma=0.98,
            exploration_final_eps=0.05,
            learning_rate=1e-2,
            buffer_size=10000,
        )


        model.learn(total_timesteps=int(timesteps), log_interval=10)
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print("Finished training!")
        model.save(path="./sb3_models/dqn_1m.txt")
        print("Eval", mean_reward, std_reward)
        self.env.close()


    def inference_sb3_dqn(self, path="./sb3_models/dqn_1m.txt", eps=10000):
        """
        Function that uses a pre-trained model and runs it in inference mode.
        """
        model = DQN(
            "CnnPolicy",
            self.env,
            verbose=1,
            learning_starts=50000,
            gamma=0.98,
            exploration_final_eps=0.02,
            learning_rate=1e-3,
            buffer_size=10000,
        )
        model.load(path=path, env=self.env)


        sum_rewards = 0
        rewards = []
        obs = self.env.reset()
        for i in range(eps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            
            sum_rewards += reward
            rewards.append(sum_rewards)
            
            self.env.render()
            if done:
                obs = self.env.reset()

        plt.figure(1)
        plt.title('Inference Mode - SB3 DQN')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.plot(rewards)
        plt.show()

        self.env.close()
