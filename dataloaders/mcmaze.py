import h5py
from torch.utils.data import Dataset


class Mcmaze(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_data()


    def load_data(self):
        # Load data from disk
        with h5py.File(self.data_dir, "r") as f:
            print(f.keys())
            self.train_behavior = f["train_behavior"][:]
            self.train_spikes_heldin = f["train_spikes_heldin"][:]
            self.train_spikes_heldout = f["train_spikes_heldout"][:]
            self.eval_spikes_heldin = f["eval_spikes_heldin"][:] if "eval_spikes_heldin" in f else None
            self.eval_spikes_heldout = f["eval_spikes_heldout"][:] if "eval_spikes_heldout" in f else None

    def __len__(self):
        return len(self.train_spikes_heldin)

    def __getitem__(self, idx):
        return (self.train_behavior[idx], self.train_spikes_heldin[idx], self.train_spikes_heldout[idx])
    

if __name__ == "__main__":
    data_dir = "datasets/mcmaze_train_bw5.h5"
    dataset = Mcmaze(data_dir)
    print(len(dataset))
    train_behavior, train_spikes_heldin, train_spikes_heldout = dataset[0]
    print(train_behavior.shape, train_spikes_heldin.shape, train_spikes_heldout.shape)
    # with h5py.File(data_dir, "r") as f:
    #     a = f['eval_spikes_heldin'][:]
    #     b = f['train_spikes_heldin'][:]

    # print((a == b).all())