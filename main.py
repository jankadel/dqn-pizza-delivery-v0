from numpy import mod
from dqn.dqn import DQNagent
from dqn.sb3_dqn import SB3DQNagent

import click

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-m', '--model', type=str, help='Select model: [DQN, SB3DQN]')
@click.option('-t', '--train', is_flag=True, help='Sets training mode')
@click.option('-i', '--infer', is_flag=True, help='Sets inference mode')
@click.option('-e', '--episodes', type=int, default=None, help='Sets number of episodes')
@click.option('-p', '--pretrained', type=str, default=None, help='Set path of pretrained model')
def main(model, train, infer, episodes, pretrained):
    if model == "DQN":
        agent = DQNagent()
        if train:
            agent.train(episodes)
        elif infer:
            if pretrained:
                agent.infer(episodes, pretrained)
            else:
                agent.infer(episodes)
    elif model == "SB3DQN":
        agent = SB3DQNagent()
        if train:
            agent.train_sb3_dqn(episodes)
        elif infer:
            if pretrained:
                agent.inference_sb3_dqn(pretrained, episodes)
            else:
                agent.inference_sb3_dqn(episodes=episodes)
    else:
        print(f"Unknown model: {model}")


if __name__ == "__main__":
    main()