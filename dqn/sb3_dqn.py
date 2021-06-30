import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt

env = gym.make("gym_pizza_delivery:gym_pizza_delivery-v0")

model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_starts=50000,
    gamma=0.98,
    exploration_final_eps=0.02,
    learning_rate=1e-3,
    buffer_size=10000,
)


model.learn(total_timesteps=int(1e6), log_interval=10)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print("Finished training!")
model.save(path="./sb3_models/dqn_1m.txt")
print("Eval", mean_reward, std_reward)

del model

model.load(path="./sb3_models/dqn_1m.txt")


sum_rewards = 0
rewards = []
obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    sum_rewards += reward
    rewards.append(sum_rewards)
    
    env.render()
    if done:
        obs = env.reset()

plt.figure(1)
plt.title('Inference Mode - SB3 DQN')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.plot(rewards)
plt.show()

env.close()
