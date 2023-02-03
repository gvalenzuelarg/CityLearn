import os
import argparse
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from tqdm import tqdm
from agents.rbc import RBC
from utils import create_environment
from replay_buffer import ReplayBuffer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates and saves replay buffers.')
    parser.add_argument('--n_steps', metavar='N', type=int, default=8760*4-1,
                        help='number of steps for the simulation. Maximum 8760*4-1 (4 years)')
    parser.add_argument('--output', metavar='O', type=str, 
                        default='output/replay_buffers.pkl',
                        help='output path.')
    args = parser.parse_args()
    n_steps = args.n_steps
    output_path = args.output
    return n_steps, output_path

def create_buffers(env):
    n_steps = env.simulation_period[1]
    replay_buffers = []
    for _, _ in enumerate(env.buildings):
        replay_buffers.append(ReplayBuffer(obs_dim=28, act_dim=3, size=n_steps))
    return replay_buffers

def main():
    n_steps, output_path = parse_arguments()

    # Instantiate CityLearn environment
    env = create_environment(n_steps)

    observations_spaces, actions_spaces = env.get_state_action_spaces()

    # Instantiating the control agent(s)
    agents = RBC(actions_spaces)

    replay_buffers = create_buffers(env)

    # Run simulation
    logging.info('Starting simulation...')
    state = env.reset()
    done = False
    for _ in tqdm(range(n_steps)):
        action = agents.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        # Save to iteration info to buffers
        for i, _ in enumerate(env.buildings):
            replay_buffers[i].store(state[i], action[i], reward[i], next_state[i], done)
    
    # Save serialized buffer
    output_dir = os.path.split(output_path)[0]
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(replay_buffers, open(output_path, 'wb'))
    logging.info('Replay buffers saved to: {}'.format(output_path))

if __name__ == '__main__':
    main()