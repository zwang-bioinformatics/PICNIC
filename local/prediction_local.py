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
        #print(self.num_layers, len(self.resblocks3d),)
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

# create model
relu=0.1
dp=0.1
model_channels = [10,10,10,10,10,   10,10,10,10,10,   10,10,10,10,10,   10,10,10,10,10,   10,10,10,10,10,   10,10,10,10,10,   10,10] # 32 conv blocks, 1 output block
device = torch.device('cuda:0')
model = ResNet(model_channels,relu, dp).eval()
model.to(device)
# ts13
model_f = f'{script_dir}/util/local_trained_model.gcn'
if not os.path.exists(model_f):
    raise FileNotFoundError
model.load_state_dict(torch.load(model_f,map_location='cuda:0'))
# print('create model',flush=True)
# CASP15 Group
AUTHOR = {'QUIC':'4898-0423-8007','PICNIC':'2613-7296-5647'}
AUTHOR_QUIC = '4898-0423-8007'
METHOD = '3D Convolutional Neural Network'

# allTargets = ['T1129s2', 'T1133', 'T1134s1', 'T1134s2', 'T1137s1', 'T1137s2', 'T1137s3', 'T1137s4', 'T1137s5', 'T1137s6', 'T1137s7', 'T1137s8', 'T1137s9', 'T1145', 'T1151s2', 'T1152', 'T1159', 'T1170', 'T1176', 'T1185s1', 'T1185s2', 'T1185s4', 'T1187', 'T1188']

df_pred_pkl = sys.argv[1]

if '/' in sys.argv[2]:
    save_dir = sys.argv[2].replace(sys.argv[2].split('/')[-1],'')
else:
    # file in cwd
    save_dir = './'


# for target in allTargets:
# print('\nTarget',target,flush=True)
# df_pred_pkls = glob.glob(f'./data-casp15/feature_TS_model/af2-*_{target}_*.pkl')
# df_pred_pkls = [i for i in df_pred_pkls if not '_out.pkl' in i]

# if len(df_pred_pkls) == 0:
#     print(f'no features of {target} at ./data-casp15/feature_TS_model/',flush=True)
#     continue

# save to 
# save_dir = f'data-casp15/refined_model/{target}/'
# if not os.path.exists(save_dir):
#     os.system(f'mkdir {save_dir}')
# data set
resolution = 0.1
# for df_pred_pkl in df_pred_pkls:
# print(f'Prediction for {df_pred_pkl}',flush=True)
# 'model_domain','idx', 'residue', 'resSeq', 'atom', 'features','atom_coord'
df_pred = pd.read_pickle(df_pred_pkl)

if '.pdb' == sys.argv[2].split("/")[-1][:-4]:
    PDB_f = f'{save_dir}{sys.argv[2].split("/")[-1].replace(".pdb","_PICNIC2-LOCAL_out.pdb")}'
elif 'local/casp15_af_models' in sys.argv[2]:
    PDB_f = f'{script_dir}/out_casp15/{sys.argv[2].split("/")[-1]+"_PICNIC2-LOCAL_out.pdb"}'
else:
    PDB_f = f'{save_dir}{sys.argv[2].split("/")[-1]+"_PICNIC2-LOCAL_out.pdb"}'
PDB_fh = open(PDB_f,'w')
PDB_TEXT = ''
clip_b = 0.025

for i,row in df_pred.iterrows():
    x = row.features
    x = x.to_dense()
    x = Variable(x).to(device, dtype=torch.float)
    x = torch.unsqueeze(x, 0)
    out = model(x)
    out = torch.squeeze(out, 0)
    out = torch.squeeze(out, 1).cpu().detach().numpy()
    new_cen = 40
#             new_cen = math.floor(1.0 / resolution)  # 16 angstrom look in each direction
    out = out[:,40-new_cen:41+new_cen,40-new_cen:41+new_cen,40-new_cen:41+new_cen]
    out_max = np.amax(out)
    max_clip_angstroms = 999999
    max_clip_cube = max_clip_angstroms / resolution
    _, out_x,out_y,out_z = np.where(out == out_max) 

    if out_max > 16: # 16 sigmoid k best so far
        # No refine if too confident
        out_x[0],out_y[0],out_z[0] = new_cen,new_cen,new_cen

    shift_cube = np.array([out_x[0],out_y[0],out_z[0]])-new_cen
    # b':0.025
    out_refine = np.clip((shift_cube)*resolution,-clip_b,clip_b)
    #print(i,out_x,out_y,out_z)
#             out_refine = (np.array([out_x[0],out_y[0],out_z[0]])-40)*0.1 # resolution

#     df_result.append([row.atom,row.residue,row.resSeq,row.atom_coord,out_refine])
    #print(np.clip(out_refine,-0.2,0.2))
#             if out_refine[0] == 0 and out_refine[1] == 0 and out_refine[2] == 0:
#                 print(f'saved_f {target} {out_refine}',flush=True)
    out_coord  = row.atom_coord + out_refine
    #print(out,out_max,torch.max(out))
    #             ATOM   4689  CE3 TRP   302      -4.324   4.177  -7.944  1.00 93.80              \n
    atom_num = ' '*(5-len(str(i+1))) + str(i+1)                   #7-11
    atom_name = row.atom +' '*(4-len(row.atom))                   #13-16
    residue_name = row.residue                                    #18-20
    residue_num = ' '*(4-len(str(row.resSeq))) + str(row.resSeq)  # 23-26

    x_coord = ' '*(8-len(f'{round(out_coord[0],3):.3f}')) + f'{round(out_coord[0],3):.3f}' # 31-38
    y_coord = ' '*(8-len(f'{round(out_coord[1],3):.3f}')) + f'{round(out_coord[1],3):.3f}' # 39-46
    z_coord = ' '*(8-len(f'{round(out_coord[2],3):.3f}')) + f'{round(out_coord[2],3):.3f}' # 47-54
    ATOM_line = 'ATOM  '+atom_num+' '+atom_name+' '+residue_name+'  '+residue_num+' '*4+x_coord+y_coord+z_coord+'  1.00  90.0           '+(atom_name if atom_name != 'CA' else 'C')
    PDB_TEXT += ATOM_line + '\n'

    #print(ATOM_line)
# df_result = pd.DataFrame(df_result,columns=['atom','residue','resSeq','atom_coord','out_refine'])
# df_result.to_pickle(df_pred_pkl.replace('.pkl','_PICNIC2_out.pkl'))
        
    # Convert .pkl file to .pdb file
    # Clip tags
#     for tag in ['a','b','c','d','e']:
#     os.system(f'python prediction_PICNIC2_clip_from_savedf.py {target} b 0')
    #os.system(f'python prediction_PICNIC2_sigmoid_from_savedf.py {target} k 0')
#     os.system(f'python prediction_PICNIC2_from_savedf.py {target}')
    

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