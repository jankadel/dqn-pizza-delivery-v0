import numpy as np
import gym

from gym import spaces
from gym_pizza_delivery.envs.delivery_2d import PizzaDelivery2D


class DeliveryEnv(gym.Env):
    """
    Wrapper class that implements the OpenAI gym interface
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self):
        """
        Function to initialize the environment
        """
        self.pizza_delivery = PizzaDelivery2D()
        self.action_space = spaces.Discrete(
            5
        )  # Five actions up,down,left,right,deliver
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.pizza_delivery.height, self.pizza_delivery.width, 3),
            dtype=np.uint8,
        )

    def step(self, action: int):
        """
        Function to let the agent take a step
        """
        ACTIONS = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
            4: "DELIVER",
        }  # Map integers to actions
        self.pizza_delivery.act(ACTIONS[action])  # act in env
        obs = self.pizza_delivery.observe()
        reward = self.pizza_delivery.evaluate()
        done = self.pizza_delivery.is_done()
        info = {}
        self.render() # Uncomment when using stable baselines
        #print(f"Reward {reward} action {ACTIONS[action]}")
        return obs, reward, done, info

    def reset(self):
        """
        Function to reset the environment which means that new customers are generated and the agent is respawned
        """
        self.close()
        self.pizza_delivery = PizzaDelivery2D()
        obs = self.pizza_delivery.observe()
        return obs

    def render(self, mode="human", close=False):
        """
        Function to return the rendered observation of the environment
        """
        if mode == "human" and close == False:
            return self.pizza_delivery.view()
        return None

    def close(self):
        """
        Function to close the environment
        """
        del self.pizza_delivery
