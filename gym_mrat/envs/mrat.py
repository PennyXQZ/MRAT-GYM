import sys
import numpy as np
import gym
from gym.utils import seeding
from gym import spaces
import enum
import h5py
from tqdm import tqdm
# from simple_trading.common.stock_market import StockMarket
# from simple_trading.common.position import ShareHolder

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

class MRAT_Env(gym.Env):
    
    # def __init__(self, num_agents):
    def __init__(self):
        super(MRAT_Env, self).__init__()
        
        self.observation_space = spaces.Box(low=0, high=100, shape=(6,))
        # self.observation_space = spaces.Dict({
        #     'SNR_5G': spaces.Box(low=0, high=100, shape=(1,)),
        #     'SNR_WiFi': spaces.Box(low=0, high=100, shape=(1,)),
        #     'SNR_LiFi': spaces.Box(low=0, high=100, shape=(1,)),
        #     'Delay_5G': spaces.Box(low=0, high=10, shape=(1,)),
        #     'Delay_WiFi': spaces.Box(low=0, high=10, shape=(1,)),
        #     'Delay_LiFi': spaces.Box(low=0, high=10, shape=(1,))
        # })

        self.action_space = spaces.Box(low=0, high=500, shape=(3,))  # Example: Discrete action space with 3 actions
  
        self.dataset_path = '../../../multipath_data.csv'
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        pass
        # response = self.ns3ZmqBridge.step(action)
        # self.envDirty = True
        return self.get_state()

    def reset(self):
        pass
        # return obs
    
    def get_random_action(self):
        act = self.action_space.sample()
        return act

    def step(self, action):
        # 1.conduct action
        # 2.get reward
        # 3.get next state
        # 4.get done
        # 5.get info
        pass

        # return obs, reward, done, info


    def get_random_action(self):
        act = self.action_space.sample()
        return act
    
    def get_dataset(self):

        data_dict = {}
        with h5py.File(self.dataset_path, 'r') as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        return data_dict

