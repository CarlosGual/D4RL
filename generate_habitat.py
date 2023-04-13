import torch
from d4rl.utils.dataset_utils import DatasetWriter
import pickle

data_writer = DatasetWriter(hidden_state=True)

with open('rollouts.pkl', 'rb') as f:
    data = pickle.load(f)

for rollout in data:
    # Rework the observations to make it consistent with D4RL
    observations = rollout.buffers['observations']
    for k in list(observations):
        # Drop unnecessary observations
        if k in ['inflection_weight', 'next_actions']:
            observations.pop(k)
            continue
        # Flatten and squeeze dimensions
        observations[k] = torch.squeeze(torch.flatten(observations[k], start_dim=0, end_dim=1))

    # Flatten the dimension associated with the parallelism in habitat
    actions = torch.squeeze(torch.flatten(rollout.buffers['actions'], start_dim=0, end_dim=1))
    rewards = torch.squeeze(torch.flatten(rollout.buffers['rewards'], start_dim=0, end_dim=1))
    dones = torch.squeeze(torch.flatten(rollout.buffers['masks'], start_dim=0, end_dim=1))
    hidden_state = torch.squeeze(rollout.recurrent_hidden_states)
    data_writer.append_data(
        observations,
        actions,
        rewards,
        dones,
        hidden_state=hidden_state
    )

data_writer.write_dataset('habitat.hdf5')
