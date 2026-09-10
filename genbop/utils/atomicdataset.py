import torch
import os
import glob
from torch_geometric.data import InMemoryDataset, Dataset


class AtomicInMemoryDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None, pre_filter=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):

        torch.save(self.collate(self.data_list), self.processed_paths[0])

class AtomicDataset(Dataset):
    def __init__(self, root, data_list=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)

    @property
    def processed_file_names(self):
        filename = os.path.join(self.processed_dir, f'data**')
        folderfile = [os.path.basename(p) for p in glob.glob(filename, recursive=True) if os.path.isfile(p)]
        return folderfile

    def process(self):
        idx = 0
        for data in self.data_list:

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
