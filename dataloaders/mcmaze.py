import h5py
from torch.utils.data import Dataset


class Mcmaze(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.load_data()

    def load_data(self):
        # Load data from disk
        with h5py.File(self.data_dir, "r") as f:
            data = f[""][:]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

if __name__ == "__main__":
    data_dir = "datasets/mcmaze_val_bw5.h5"
    with h5py.File(data_dir, "r") as f:
        a = f['eval_spikes_heldin'][:]
        b = f['train_spikes_heldin'][:]

    print((a == b).all())