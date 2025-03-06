import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from copy import deepcopy
import torch
import os
from torch.nn import LazyLinear
import torch.nn.functional as F
import torch.optim as optim
import pickle

from utils import trunc_normal
import time

from IPython.display import clear_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pvt_graph import PVTGraph

class ReplayBuffer:
    """
    Algorithm: Experience Replay Buffer with PVT Corner Support
    Features:
    - Separate buffers for each PVT corner
    - Efficient storage and sampling mechanisms
    - Support for attention weights and rewards
    """
    def __init__(self, CktGraph, PVT_Graph, size: int, batch_size: int = 32):
        # Initialize buffer parameters
        self._init_buffer_params(CktGraph, PVT_Graph, size, batch_size)
        self._init_corner_buffer()

    def _init_corner_buffer(self):
        """
        Algorithm: Corner Buffer Initialization
        Process:
        1. Create separate buffer for each PVT corner
        2. Initialize storage arrays for:
           - Observations and next states
           - Actions and rewards
           - PVT states and transitions
           - Attention weights and indices
        """
        # Pseudo-implementation
        self._create_corner_buffers()

    def store(
        self,
        pvt_state: np.ndarray,
        action: np.ndarray,
        results_dict: dict,
        next_pvt_state: np.ndarray,
        corner_indices: list,
        attention_weights: np.ndarray,
        total_reward: float,
        done: bool,
    ):
        """
        Algorithm: Experience Storage
        Process:
        1. Preprocess attention weights
        2. Store transition for each corner:
           - Update observation and state information
           - Store action and reward data
           - Record PVT state transitions
           - Update buffer pointers
        """
        # Pseudo-implementation
        self._preprocess_attention_weights(attention_weights)
        self._store_transitions(pvt_state, action, results_dict, next_pvt_state,
                              corner_indices, attention_weights, total_reward, done)

    def sample_corner_batch(self, corner_idx: int) -> Dict[str, np.ndarray]:
        """
        Algorithm: Corner-specific Batch Sampling
        Process:
        1. Validate corner index and buffer size
        2. Random sampling without replacement
        3. Return batch with:
           - State-action pairs
           - Rewards and next states
           - PVT information and attention data
        """
        # Pseudo-implementation
        return self._sample_batch(corner_idx)


class DDPGAgent:
    def __init__(
        self,
        env,
        CktGraph,
        PVT_Graph,
        Actor,
        Critic,
        memory_size: int,
        batch_size: int,
        noise_sigma: float,
        noise_sigma_min: float,
        noise_sigma_decay: float,
        noise_type: str,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        sample_num: int = 3,
        agent_folder: str = None,
        old = False
    ):
        """
        Algorithm: DDPG Agent Initialization
        Components:
        1. Actor-Critic Networks:
           - Policy network (Actor)
           - Value network (Critic)
           - Target networks
        2. Experience Replay:
           - PVT corner-specific buffers
           - Prioritized sampling
        3. Exploration Strategy:
           - Adaptive noise parameters
           - Multiple noise types
        """
        # Pseudo-implementation
        self._init_networks(env, CktGraph, PVT_Graph, Actor, Critic)
        self._init_memory_and_parameters(memory_size, batch_size, noise_params)
        self._setup_training_components()

    def _normalize_pvt_graph_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Algorithm: PVT Graph State Normalization
        Process:
        1. Clone input state
        2. Apply specific normalization for:
           - Voltage and temperature
           - Performance metrics
           - Circuit parameters
        3. Scale values to appropriate ranges
        """
        # Pseudo-implementation
        return self._apply_normalization(state.clone())

    def load_agent(self, agent_folder):
        """
        Algorithm: Agent Loading
        Process:
        1. Load network weights:
           - Actor and target actor
           - Critic and target critic
        2. Load replay buffer state
        3. Restore training parameters
        """
        # Pseudo-implementation
        self._load_network_weights(agent_folder)
        self._load_replay_buffer(agent_folder)
        self._restore_training_state(agent_folder)