from d4rl.utils.dataset_utils import DatasetWriter
import pickle

data_writer = DatasetWriter()

with open('rollouts.pkl', 'rb') as f:
    data = pickle.load(f)

for update in data:

    data_writer.append_data()
