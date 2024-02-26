import sys

assert len(sys.argv) == 3, f'Usage: python {sys.argv[0]} <features_path> <original_pdb_path>'

import glob
import numpy as np
import pickle
import os
import datetime
import math
from math import sqrt
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import time
import pandas as pd

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
script_dir = os.path.dirname(os.path.abspath(__file__))

class ResNet(nn.Module):
    def __init__(self, channels,relu=0.1, dp=0.1):
        super(ResNet, self).__init__()
        self.num_layers = len(channels) - 1
        self.ks = 5
        self.dv = 1
        self.resblocks3d = nn.ModuleList()
        self.relu = nn.LeakyReLU(relu)
        for i in range(self.num_layers):
            #print('i',i,channels[i], channels[i+1])
            layer_conv3d = torch.nn.Conv3d(channels[i], channels[i+1], kernel_size=self.ks, stride=1,
                                           padding=int(self.dv*(self.ks-1)/2), dilation=self.dv)
            nn.init.xavier_uniform_(layer_conv3d.weight, gain=sqrt(2.0))
            # append layers
            self.resblocks3d.append(layer_conv3d)
            self.resblocks3d.append(nn.BatchNorm3d(channels[i+1], affine=True))
            self.resblocks3d.append(nn.LeakyReLU(relu))
            self.resblocks3d.append(nn.Dropout(p=dp))
        self.out = torch.nn.Conv3d(channels[i+1],1, kernel_size=self.ks, stride=1,
                                           padding=int(self.dv*(self.ks-1)/2), dilation=self.dv)
        nn.init.xavier_uniform_(self.out.weight, gain=sqrt(2.0))
        self.norm = nn.BatchNorm3d(1, affine=True)
    def forward(self, x):
        for layer_i in range(0,self.num_layers,1):
            i = layer_i*4
            # block 1
            residual = x # make pointer to data stored at x
            x = self.resblocks3d[i](x) # x now points to different data, residual still points to old x
            x = self.resblocks3d[i+1](x)
            x = self.resblocks3d[i+2](x)
            x = self.resblocks3d[i+3](x)
            x = x + residual # residual skip connection
        x = self.norm(self.out(x))
        return x

def clip(x,arr_length):
    return np.clip(x,0,arr_length-1)
# create model
relu=0.1
dp=0.1
model_channels = [10] * 32
# [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10] # 32 conv blocks, 1 output block
device = torch.device('cuda:0')
model = ResNet(model_channels,relu, dp).eval()
model.to(device)
model_f = f'{script_dir}/util/global_trained_model.gcn'
if not os.path.exists(model_f):
    raise FileNotFoundError
model.load_state_dict(torch.load(model_f,map_location='cuda:0'))
# print('create model',flush=True)
# CASP15 Group
AUTHOR = {'QUIC':'4898-0423-8007','PICNIC':'2613-7296-5647'}
AUTHOR_QUIC = '4898-0423-8007'
METHOD = '3D Residual Neural Network'

# casp15 example
# df_pred_pkls_path = './casp15_only_targets_global_test_data_sparse/*.pkl'
# df_pred_pkls = glob.glob(df_pred_pkls_path)
df_pred_pkl = sys.argv[1]

# if len(df_pred_pkls) == 0:
#     print(f'no features at ./var/features.pkl',flush=True)

tag = 'w'
clip_thresholds = {tag:0.0125}
clip_count = {tag:[]}
tag_thresholds = {tag:0.0125}
tag_count = {tag:[]}
clip_or_sigmoid = 'clip'
center_z_shift = 0
use_only_casp15 = True

# improved_list = []
# tmscore_improvements = []
# af_tmscore_avg,refined_model_tmscore_avg,tmscore_count = 0,0,0
# num_changed,num_unchanged = 0,0
# avg_percent_center = 0
# for df_pred_index,df_pred_pkl in enumerate(df_pred_pkls):
#     if df_pred_index % 100 == 0:
#         print(f'{100*df_pred_index / len(df_pred_pkls):.1f}% complete',flush=True)
    
# save to 
# if use_only_casp15:
#     if df_pred_pkl.split('/')[-1][:4] != 'af2-':
#         continue
#     target = df_pred_pkl.split('/')[-1].split('_')[1]
#     save_dir = f'output/{target}/'
#     else: 
#         target = df_pred_pkl.split('-')[1]
#         save_dir = f'refined_cell_global_model_uniprot_test_targets/{target}/'
# if not os.path.exists(save_dir):
#     os.system(f'mkdir {save_dir}')
# data set
if '/' in sys.argv[2]:
    save_dir = sys.argv[2].replace(sys.argv[2].split('/')[-1],'')
else:
    # file in cwd
    save_dir = './'
#         print(f'Prediction for {df_pred_pkl}',flush=True)
df_pred = pd.read_pickle(df_pred_pkl)
features_name = 'features'
x_dim, y_dim, z_dim = df_pred[features_name].iloc[0].size()[1:]
if x_dim * y_dim * z_dim > 9000000:
    print('ERROR: protein too large',df_pred_pkl,x_dim * y_dim * z_dim,flush=True)
    raise Exception

# Cuboid of features
xs = torch.stack(tuple(df_pred[features_name])).to_dense()

xs = Variable(xs).to(device, dtype=torch.float)
out = model(xs)
del xs
out = torch.squeeze(out, 0)
out = torch.squeeze(out, 1).cpu().detach().numpy()
new_cuboid = out[0]
del out

df_pred_atom_coord = df_pred.atom_coord.iloc[0]
df_result = []
model_num = df_pred_pkl.replace('.pkl','').split('_')[-1]
PDB_TEXT = ''

# Can refine atoms beyond the bounds of original protein
cuboid_search_radius = 4
resolution = 0.6
# Side lengths of cuboid (not necessarily equal)
min_x, min_y, min_z = round(min(np.array(df_pred_atom_coord['CA'])[:,0])/resolution), round(min(np.array(df_pred_atom_coord['CA'])[:,1])/resolution), round(min(np.array(df_pred_atom_coord['CA'])[:,2])/resolution)
x_length, y_length, z_length = round(max(np.array(df_pred_atom_coord['CA'])[:,0])/resolution) - min_x + 1 + 2*cuboid_search_radius, round(max(np.array(df_pred_atom_coord['CA'])[:,1])/resolution) - min_y + 1 + 2*cuboid_search_radius, round(max(np.array(df_pred_atom_coord['CA'])[:,2])/resolution) - min_z + 1 + 2*cuboid_search_radius
xyz_cuboid_mins = [min_x,min_y,min_z]
# AF CA coord
if np.isnan(new_cuboid).any():
    print("The array contains NaN values.",flush=True)
new_cuboid_max = np.amax(new_cuboid)

if '.pdb' == sys.argv[2].split("/")[-1][:-4]:
    PDB_f = f'{save_dir}{sys.argv[2].split("/")[-1].replace(".pdb","_PICNIC2-GLOBAL_out.pdb")}'
else:
    PDB_f = f'{save_dir}{sys.argv[2].split("/")[-1]+"_PICNIC2-GLOBAL_out.pdb"}'

PDB_fh = open(PDB_f,'w')
PDB_TEXT = ''

for residue_index,atom_coord in enumerate(df_pred_atom_coord['CA']):
    '''
    Normalize in cuboid: Scale with resolution and subtract cuboid mins to align lowest with index 0.
    Add search radius to allow for buffer space when refining.
    '''
    af_atom_coord_in_cuboid = (np.array(np.round(atom_coord/resolution))-\
                        np.array(xyz_cuboid_mins) + np.array([cuboid_search_radius,cuboid_search_radius,cuboid_search_radius])).astype(int)

    # Find highest value within certain range
    max_clip_angstroms = 1.6
    max_clip_cubic_index = math.floor(max_clip_angstroms / resolution) # In this case, 10
    x_cen,y_cen,z_cen = af_atom_coord_in_cuboid
    search_new_cubic = new_cuboid[clip(x_cen-max_clip_cubic_index,new_cuboid.shape[0]) : clip(x_cen+max_clip_cubic_index+1,new_cuboid.shape[0]), clip(y_cen-max_clip_cubic_index,new_cuboid.shape[1]) : clip(y_cen+max_clip_cubic_index+1,new_cuboid.shape[1]), clip(z_cen-max_clip_cubic_index,new_cuboid.shape[2]) : clip(z_cen+max_clip_cubic_index+1,new_cuboid.shape[2])]

    search_new_cubic_max = np.amax(search_new_cubic)
    old_cubic_max = search_new_cubic_max
    search_x,search_y,search_z = np.where(search_new_cubic == search_new_cubic_max)
    if search_x[0] == max_clip_cubic_index and search_y[0] == max_clip_cubic_index and search_z[0] == max_clip_cubic_index+center_z_shift:
        # Don't allow refinement to stay in the center
        search_new_cubic[max_clip_cubic_index,max_clip_cubic_index, max_clip_cubic_index+center_z_shift] = -9999999
        search_new_cubic_max = np.amax(search_new_cubic)
        search_x,search_y,search_z = np.where(search_new_cubic == search_new_cubic_max)
    if search_new_cubic_max / old_cubic_max >= 0.93:
        shift = np.array([search_new_cubic.shape[0]//2, search_new_cubic.shape[1]//2, search_new_cubic.shape[2]//2])
        # Move x,y,z by search_(x,y,z) values * resolution
        search = (np.array([search_x[0],search_y[0],search_z[0]])-shift)*resolution # Move by 0.2
#         num_changed += 1
    else:
        # Keep original atom location as there is no confident answer
        search = np.array([0,0,0])
#         num_unchanged += 1
#     avg_percent_center += search_new_cubic_max / old_cubic_max

    # Other atoms in same residue shift same amount
    for atom in ['N','CA','C','O']:
        af_atom_coord = df_pred_atom_coord[atom][residue_index]
        # New coordinates
        if 'clip' in clip_or_sigmoid:
            scaled_refine = np.clip(search,-tag_thresholds[tag],tag_thresholds[tag])
        else:
            raise Exception
        out_coord = af_atom_coord + scaled_refine
        atom_num = ' '*(5-len(str(int(df_pred.resSeq.iloc[0][residue_index])+1))) + str(int(df_pred.resSeq.iloc[0][residue_index])+1)                   #7-11
        atom_name = ' '+atom +' '*(3-len(atom))                   #13-16
        residue_name = df_pred.residue.iloc[0][residue_index]                                    #18-20
        residue_num = ' '*(4-len(str(df_pred.resSeq.iloc[0][residue_index]))) + str(df_pred.resSeq.iloc[0][residue_index])  # 23-26

        x_coord = ' '*(8-len(f'{round(out_coord[0],3):.3f}')) + f'{round(out_coord[0],3):.3f}' # 31-38
        y_coord = ' '*(8-len(f'{round(out_coord[1],3):.3f}')) + f'{round(out_coord[1],3):.3f}' # 39-46
        z_coord = ' '*(8-len(f'{round(out_coord[2],3):.3f}')) + f'{round(out_coord[2],3):.3f}' # 47-54
        ATOM_line = 'ATOM  '+atom_num+' '+atom_name+' '+residue_name+'  '+residue_num+' '*4+x_coord+y_coord+z_coord+'  1.00  900            '+(atom if atom != 'CA' else 'C')
        PDB_TEXT += ATOM_line + '\n'
del new_cuboid
# os.system(f'cp {sys.argv[2]} PDB_f')
with open(sys.argv[2],'r') as input_pdb:
    input_lines = input_pdb.readlines()

out_lines = PDB_TEXT.split('\n')
if len(out_lines[-1]) == 0:
    del out_lines[-1]
# atom name, residue number, x, y, z coords
out_change = [[line[12:16],line[22:27],line[30:38],line[38:46],line[46:54]] for line in out_lines]
out_atom = out_change.pop(0)

out_pdb_lines = []
for line in input_lines:
#     print(len(out_change))
    if line[12:16].replace(' ','') == out_atom[0].replace(' ','') and line[22:27].replace(' ','') == out_atom[1].replace(' ',''):
        # Update coords if atoms and residues match
        line_list = list(line)
        line_list[30:38],line_list[38:46],line_list[46:54] = out_atom[2],out_atom[3],out_atom[4]
        line = ''.join(line_list)
        if len(out_change) != 0:
            out_atom = out_change.pop(0)
    out_pdb_lines.append(line)
assert len(out_change) == 0, 'PDB update failed'
PDB_TEXT = ''.join(out_pdb_lines)
PDB_fh.write(PDB_TEXT)
PDB_fh.close()
print(PDB_f)