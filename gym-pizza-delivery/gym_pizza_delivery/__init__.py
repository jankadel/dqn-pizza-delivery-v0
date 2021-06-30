from gym.envs.registration import register

register(
    id='gym_pizza_delivery-v0',
    entry_point='gym_pizza_delivery.envs:DeliveryEnv',
)