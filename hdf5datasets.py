import h5py
from torch.utils.data import Dataset, ConcatDataset
import glob


class HDF5EoSDataset(Dataset):
    """
    Dataset class for hdf5 files configured to provide masses
    and tidal deformabilities together with the concatenated
    model strains projected on LIGO and Virgo detectors.
    """

    def __init__(self, h5path):
        self.h5path = h5path

    def __getitem__(self, index):
        with h5py.File(self.h5path, "r") as file:

            dataset_name = "merger_BNS_" + str(index)
            dataset = file[dataset_name]
            waveforms = dataset[:]

            mass_1 = dataset.attrs["mass_1"]
            mass_2 = dataset.attrs["mass_2"]
            lambda_1 = dataset.attrs["lambda_1"]
            lambda_2 = dataset.attrs["lambda_2"]

            return waveforms, mass_1, mass_2, lambda_1, lambda_2

    def __len__(self):
        with h5py.File(self.h5path, 'r') as file:
            return len(file.keys())


def merge_trainsets(path_dir):

    datasets = []

    for filepath in glob.iglob(path_dir):

        train_dataset = HDF5EoSDataset(filepath)
        datasets.append(train_dataset)

    merged_trainset = ConcatDataset(datasets)

    return merged_trainset
