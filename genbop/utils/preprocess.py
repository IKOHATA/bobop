from ase.io import read
from .atomicdata import AtomicData
from .atomicdataset import AtomicInMemoryDataset
from ase.io.extxyz import write_extxyz
import numpy as np

def create_datasets(datafile,outdir,cutoff,filetype='extxyz'):

    rcut = cutoff
    if type(datafile) is str:
        filenames = [datafile]
    elif type(datafile) is list:
        filenames = datafile

    datalist_all=[]
    for filename in filenames:
        structures=read(filename,index=':')
        datalist = [AtomicData.from_atoms(atoms,rcut) for atoms in structures]
        datalist_all.extend(datalist)

    dataset=AtomicInMemoryDataset(outdir, datalist_all)

    return dataset

def split(datafile,trainfile,validfile,valid_frac=0.1):

    structures = read(datafile,index=':')

    arr = np.arange(len(structures))

    np.random.shuffle(arr)

    split_idx = int(len(arr) * valid_frac)

    valid = [structures[i] for i in arr[:split_idx]]
    train = [structures[i] for i in arr[split_idx:]]

    write_extxyz(trainfile,train)
    write_extxyz(validfile,valid)





