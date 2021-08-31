import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.env_util import make_vec_env
#env = make_vec_env("gym_pizza_delivery:gym_pizza_delivery-v0", n_envs=4)

env = gym.make("gym_pizza_delivery:gym_pizza_delivery-v0")


# Code snippet that uses a stable implementation of PPO by stable_baselines3 (see: https://stable-baselines3.readthedocs.io/en/master/) 
# This version of PPO is used as a benchmark for the own implementation of DQN - However it was not further pursued because PPO did not seem
# to work properly on the environment "gym_pizza_delivery-v0"

model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1,
    gamma=0.99,
    learning_rate=1e-4,
    n_steps=2048,
    
)

model.learn(total_timesteps=int(1.5e6), log_interval=10)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("Finished training!")
model.save(path="./sb3_models/ppo.txt")
print("Eval", mean_reward, std_reward)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()


env.close()