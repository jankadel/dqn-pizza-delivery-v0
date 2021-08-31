import pygame
import random
import copy


class DeliveryAgent:
    """
    Class that defines the delivery agent and its behavior.
    """
    def __init__(self, agent_file, map_file, pos, screen_width, screen_height):
        self.map = pygame.image.load(map_file)
        self.agent = pygame.image.load(agent_file)
        self.agent = pygame.transform.scale(self.agent, (21, 21))
        self.screen_h = screen_height
        self.screen_w = screen_width

        self.agent_col = (246, 255, 0, 255) # Agent color (yellow)
        self.non_road_col = (14, 135, 11, 255) # Non-road color (green)
        self.road_col = (122, 122, 122, 255) # Road color (grey)
        self.start_col = (0, 70, 191, 255) # Start color #0b4ab0 or rgb(11, 74, 176) or blue

        self.start_pos = copy.deepcopy(pos)
        self.pos = pos
        self.center = [self.pos[0] + 11, self.pos[1] + 11]
        self.colors = [
          (0, 255, 225, 255), 
          (0, 255, 94, 255), 
          (255, 0, 0, 255), 
          (255, 145, 0, 255), 
          (255, 0, 170, 255)
        ] # Cyan, light green, red, orange, pink
        self.customers = list() # List of customers - ((Position), Color)
        self.num_customers = len(self.colors)
        self.last_delivered = None

        self.successful_deliveries = 0
        self.goal_spawned = False
        #self.spawn_goal()
        self.time_spent = 0  # Time spent for delivery since delivery should not take forever...

    def update(self, delivery):
        """
        Updates the game state (agent.center, delivered customers)
        """
        self.center = [self.pos[0] + 11, self.pos[1] + 11]
        self.time_spent += 1
        if delivery[0]:
            if delivery[1]:
                for c in self.customers:
                    if delivery[0] == c[1]:
                        self.last_delivered = self.customers[self.customers.index(c)]
                        self.customers[
                            self.customers.index(c)
                        ] = "DELIVERED"  # Overwrite the customer that was served
                        self.successful_deliveries += 1
                        print("BUG", self.successful_deliveries, self.num_customers)
            else:
                self.customers.append("FAILED")

        if self.successful_deliveries == self.num_customers:
            self.successful_deliveries = 0 # reset such that goal is only spawned once!
            self.spawn_goal()

    def draw_agent(self, screen):
        """
        Draws the agent
        """
        screen.blit(self.agent, self.pos)

    def draw_customers(self):
        """
        Draws customers from self.customers
        """
        for c in self.customers:
            #print("Customer", c)
            pygame.draw.rect(self.map, c[1], (c[0][0], c[0][1], 22, 22))

    def delete_delivered_customer(self):
        """
        Deletes the customer that was previously served.
        """
        if self.last_delivered:
            #print(self.last_delivered)
            pygame.draw.rect(self.map, self.road_col, (self.last_delivered[0][0], self.last_delivered[0][1], 22, 22))
            self.last_delivered = None

    def generate_random_customers(self):
        """
        Generates self.num_customers many, randomly placed customers ((pos), Color)
        """
        customer_colors = copy.deepcopy(self.colors)
        while len(self.customers) < self.num_customers:
            x = random.randint(0, self.screen_w)
            y = random.randint(0, self.screen_h)

            if x + 22 < self.screen_w and y + 22 < self.screen_h:
                tl = self.get_ground_color([x, y])
                br = self.get_ground_color([x + 21, y + 21])
                tr = self.get_ground_color([x + 21, y])
                bl = self.get_ground_color([x, y + 21])
                center = self.get_ground_color([x + 11,y + 11])
                c = self.road_col

                if center != c or tl != c or br != c or tr != c or bl != c:
                    continue
                else:
                   self.customers.append([[x,y], customer_colors[0]])
                   customer_colors.pop(0)

    def get_ground_color(self, pos=None):
        """
        Returns the color of the map below the center of the agent
        """
        if pos:
            col = pygame.Surface.get_at(self.map, pos)
        else:
            #print("P,C", self.pos, self.center)
            col = pygame.Surface.get_at(self.map, self.center)

        return col

    def check_collision(self):
        """
        Checks if the agent collided with non-road area
        """
        top = [self.pos[0], self.pos[1]]
        bot = [self.pos[0], self.pos[1] + 21]
        left = [self.pos[0] + 21, self.pos[1]]
        right = [self.pos[0] + 21, self.pos[1] + 21]

        top_c = self.get_ground_color(top)
        bot_c = self.get_ground_color(bot)
        left_c = self.get_ground_color(left)
        right_c = self.get_ground_color(right)

        # print("Cols:", top_c, bot_c, left_c, right_c)
        c = self.non_road_col
        if c == top_c or c == bot_c or c == left_c or c == right_c:
            return True
        else:
            return False

    def check_delivery(self):
        """
        Checks if the agent is able to deliver
        """
        top = [self.pos[0], self.pos[1]]
        bot = [self.pos[0], self.pos[1] + 21]
        left = [self.pos[0] + 21, self.pos[1]]
        right = [self.pos[0] + 21, self.pos[1] + 21]

        top_c = self.get_ground_color(top)
        bot_c = self.get_ground_color(bot)
        left_c = self.get_ground_color(left)
        right_c = self.get_ground_color(right)

        # print("Cols:", top_c, bot_c, left_c, right_c)
        if top_c in self.colors:
            return top_c
        elif bot_c in self.colors:
            return bot_c
        elif left_c in self.colors:
            return left_c
        elif right_c in self.colors:
            return right_c
        else:
            return None


    def spawn_goal(self):
        """
        Spawns the goal
        """
        self.goal_spawned = True
        pygame.draw.rect(self.map, self.start_col, (self.start_pos[0], self.start_pos[1], 23, 23))
