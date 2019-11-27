import scipy.interpolate
from PIL import Image
import numpy as np
import glob
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
STACK_NPZ_FILENAME_DH = os.path.join(SRC_DIR, 'dh_zstack.npz')
DATA_DIR_DH = "data/sequence-as-stack-Beads-DH-Exp-as-list/"

STACK_NPZ_FILENAME_AS = os.path.join(SRC_DIR, 'as_zstack.npz')
DATA_DIR_AS = "data/z-stack-Beads-AS-Exp-as-list/"

STACK_NPZ_FILENAME_2D = os.path.join(SRC_DIR, '2d_zstack.npz')
DATA_DIR_2D = "data/z-stack-Beads-2D-Exp-as-list/"

def load_2D():
    return dict(np.load(STACK_NPZ_FILENAME_2D))

def load_AS():
    # the NPZ object can't be pickled causing all manner of hell
    # with pywren, so we convert it to a dict
    return dict(np.load(STACK_NPZ_FILENAME_AS).items())

def load_DH():
    return dict(np.load(STACK_NPZ_FILENAME_DH).items())

def load_name(n):
    if n == 'emp_as':
        return load_AS()
    elif n == 'emp_dh':
        return load_DH()
    elif n == 'emp_2d':
        return load_2D()
    else:
        raise ValueError()


# z_dict = load_name('emp_as')
# print(z_dict['imgs'].shape)
# print(z_dict['z_vals'].shape)
# print(z_dict['z_stack'].shape)
# activations = np.loadtxt("data/activations.csv", delimiter=",")
# print(activations.shape)


print(np.linspace(0.0, 64, 6400).shape)