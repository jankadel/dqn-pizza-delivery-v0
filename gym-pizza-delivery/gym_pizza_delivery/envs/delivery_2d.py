import os
import pygame
from PIL import Image
import numpy as np
from gym_pizza_delivery.envs.delivery_agent import DeliveryAgent


class PizzaDelivery2D:
    def __init__(
        self,
        agent_file: str = None,
        map_file: str = None,
        resource_path=None,
        screen_width: int = 500,
        screen_height: int = 500,
    ) -> None:
        resource_path = (
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
            if resource_path is None
            else resource_path
        )
        agent_file = "agent.png" if agent_file is None else agent_file
        map_file = "delivery_map.png" if map_file is None else map_file
        pygame.init()
        pygame.display.set_caption("Pizza Delivery: Papa Jan's")
        self.width = screen_width
        self.height = screen_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.agent = DeliveryAgent(
            agent_file=os.path.join(resource_path, agent_file),
            map_file=os.path.join(resource_path, map_file),
            pos=[228, 452],
            screen_width=screen_width,
            screen_height=screen_height,
        )
        self.game_speed = 60
        self.agent.generate_random_customers()
        self.done = False

    def act(self, action: str):
        """
        Makes the agent perform an action {UP, DOWN, LEFT, RIGHT, DELIVER}
        """
        delivery = (None, False)
        if action == "DOWN":
            if self.agent.pos[1] < self.height - 25:
                self.agent.pos[1] += 3
        elif action == "UP":
            if self.agent.pos[1] > 2:
                self.agent.pos[1] -= 3
        elif action == "LEFT":
            if self.agent.pos[0] > 2:
                self.agent.pos[0] -= 3
        elif action == "RIGHT":
            if self.agent.pos[0] < self.width - 25:
                self.agent.pos[0] += 3
        elif action == "DELIVER":
            #g_col = self.agent.check_delivery()
            g_col = self.agent.get_ground_color()
            if g_col in self.agent.colors:
                delivery = (g_col, True)
                print(action, delivery)
            else:
                delivery = (g_col, False)
        else:
            raise ValueError(f"{action} is not a valid action.")

        # print("Position:", self.agent.pos)
        self.agent.update(delivery)

    def evaluate(self):
        """
        Evaluates the current state of the agent and calculates the reward on basis of the state
        """
        if "FAILED" in self.agent.customers:
            reward = -100
            self.agent.customers.remove("FAILED")
        elif "DELIVERED" in self.agent.customers:
            reward = 10000
            self.agent.customers.remove("DELIVERED")
        elif self.agent.goal_spawned == True and self.agent.get_ground_color() == self.agent.start_col:
            reward = 10000
        else:
            if self.agent.check_collision():
                reward = -50
            else:
                reward = -1
        return reward

    def is_done(self):
        """
        Checks if the termination condition is satisfied and returns true or false based on the condition
        """
        if self.agent.get_ground_color() == self.agent.start_col:
            if len(self.agent.customers) == 0:
                self.done = True
                return self.done
        return self.done

    def observe(self):
        """
        Returns an image of the environment state
        """
        surface = pygame.image.tostring(self.screen, "RGB")
        img = Image.frombytes(
            "RGB", (self.width, self.height), surface
        )  # raw byte string
        img = np.asarray(img, dtype="int32")
        return img

    def view(self):
        """
        Update the view
        """
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        self.screen.blit(self.agent.map, (0, 0))

        self.agent.draw_agent(self.screen)
        self.agent.delete_delivered_customer()
        self.agent.draw_customers()

        pygame.display.flip()
        self.clock.tick(self.game_speed)
        return self.screen