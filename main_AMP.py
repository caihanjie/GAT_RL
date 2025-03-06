import torch
import numpy as np
import os
import gymnasium as gym
import multiprocessing

import pickle  

from ckt_graphs import GraphAMPNMCF
from ddpg import DDPGAgent
from datetime import datetime

from utils import ActionNormalizer, OutputParser2
from models import ActorCriticPVTGAT
from AMP_NMCF import AMPNMCFEnv
from pvt_graph import PVTGraph

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    date = datetime.today().strftime('%Y-%m-%d')
    PWD = os.getcwd()
    SPICE_NETLIST_DIR = f'{PWD}/simulations'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    pvtGraph = PVTGraph()
    pvtGraph._clean_pvt_dirs()
    pvtGraph._create_pvt_dirs()
    pvtGraph._create_pvt_netlists()
    pvtGraph._create_pvt_netlists_tran()
    CktGraph = GraphAMPNMCF
    GNN = ActorCriticPVTGAT

    continue_training = False
    laststeps = 0
    old = False
    agent_folder = './saved_results/saved_0.81'

    load_buffer = False
    load_buffer_size = 18
    buffer_path = './saved_results/02-01_23-44_steps18_corners-5_reward--3.39/memory_02-01_23-44_steps18_corners-5_reward--3.39.pkl'  

    plot_interval = 1
    print_interval = 3

    sample_num = 2
    num_steps = 10
    initial_random_steps = 2
    batch_size = 2
    
    check_interval = 4
    noise_sigma = 0.05
    noise_sigma_min = 0.05
    noise_sigma_decay = 0.9995
    noise_type = 'uniform' 
    THREAD_NUM = 2
    memory_size = laststeps + num_steps+ 10 + load_buffer_size

    run_intial = False
    if run_intial == True:
        env = AMPNMCFEnv()
        env._init_random_sim(100)
        
    from gymnasium.envs.registration import register

    env_id = 'sky130AMP_NMCF-v0'
    env_dict = gym.envs.registration.registry.copy()

    for env in env_dict:
        if env_id in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry[env]

    print("Register the environment")
    register(
            id = env_id,
            entry_point = 'AMP_NMCF:AMPNMCFEnv',
            max_episode_steps = 5,
            kwargs={'THREAD_NUM': THREAD_NUM ,'print_interval':print_interval}
            )
    env = gym.make(env_id)  
                
    agent = DDPGAgent(
        env, 
        CktGraph(),
        PVTGraph(),
        GNN().Actor(CktGraph(), PVTGraph()),
        GNN().Critic(CktGraph()),
        memory_size, 
        batch_size,
        noise_sigma,
        noise_sigma_min,
        noise_sigma_decay,
        initial_random_steps=initial_random_steps,
        noise_type=noise_type, 
        sample_num=sample_num,
        agent_folder= agent_folder,
        old = old
    )
    
    if load_buffer == True:
        agent.load_replay_buffer(buffer_path)
    
    agent.train(num_steps, plot_interval ,check_interval , continue_training=continue_training)

    print("********Replay the best results********")
    memory = agent.memory
    best_reward = float('-inf')
    best_action = None
    best_corner = None

    for corner_idx, buffer in memory.corner_buffers.items():
        rewards = buffer['total_reward'][:buffer['size']]
        if len(rewards) > 0:
            max_reward = np.max(rewards)
            if max_reward > best_reward:
                best_reward = max_reward
                idx = np.argmax(rewards)
                best_action = buffer['action'][idx]
                best_corner = corner_idx

    if best_action is not None:
        results_dict, flag, terminated, truncated, info = agent.env.step(
            (best_action, np.arange(pvtGraph.num_corners), True)
        )
        
    save = True
    if save == True:
        num_steps = agent.total_step
        current_time = datetime.now().strftime('%m-%d_%H-%M')
        folder_name = f"{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}"
        save_dir = os.path.join(PWD, 'saved_results', folder_name)
        
        os.makedirs(save_dir, exist_ok=True)

        results_file_name = f"opt_result_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}"
        results_path = os.path.join(save_dir, results_file_name)
        with open(results_path, 'w') as f:
            f.writelines(agent.env.unwrapped.get_saved_results)  

        model_weight_actor = agent.actor.state_dict()
        save_name_actor = f"Actor_weight_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pth"
        
        model_weight_critic = agent.critic.state_dict()
        save_name_critic = f"Critic_weight_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pth"
        
        torch.save(model_weight_actor, os.path.join(save_dir, save_name_actor))
        torch.save(model_weight_critic, os.path.join(save_dir, save_name_critic))
        print("Actor and Critic weights have been saved!")

        memory_path = os.path.join(save_dir, f'memory_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pkl')
        with open(memory_path, 'wb') as memory_file:
            pickle.dump(memory, memory_file)

        agent_path = os.path.join(save_dir, f'DDPGAgent_{current_time}_steps{num_steps}_corners-{pvtGraph.num_corners}_reward-{best_reward:.2f}.pkl')
        with open(agent_path, 'wb') as agent_file:
            pickle.dump(agent, agent_file)
            
        print(f"All results have been saved in: {save_dir}")


