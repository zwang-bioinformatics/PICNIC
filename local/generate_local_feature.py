import glob
import os,sys
import pandas as pd 
#print('pandas version',pd.__version__)
# from utils import *
import numpy as np
import math
import torch
from scipy.spatial import distance_matrix
from Bio import SeqIO
from Bio.SeqUtils import seq3,seq1

# server_dir = './'
# unpress_dir = f'{server_dir}/data-casp15/unpress_TS_model/'
# featrue_dir = f'{server_dir}/data-casp15/feature_TS_model/'
# mass_law_casp14_dir = '../MASS-CASP14_LAW-CASP14_Server/'
script_dir = os.path.dirname(os.path.abspath(__file__))
mass_law_casp14_job_dir = f'{script_dir}/util/mass_law_casp15/'
temp_dir = f'{script_dir}/.var/'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


def get_pdb_atom(file_pdb,atom_):
    coords, chains, atomNames, residues, resSeqs, chainIDs = [], [], [], [], [],[]
    fileLines = open(file_pdb)
    alternate_indicator_keep_max = {}
    for line in fileLines:
        line2 = line.rstrip()
        if line2[0:4] == 'ATOM':
            atomName = line2[12:16]
            atomName = atomName.replace(" ", "")
            resName = line2[17:20]
            if True:
                #if (resName == 'GLY' and atomName == 'CA') or (resName != 'GLY' and atomName == 'CB'):
                if atomName == atom_:
                    atomNames.append(atomName)
                    alternate_indicator = line2[16]
                    chainID = line2[21:22]
                    resSeq = line2[22:27]
                    resSeq = resSeq.replace(" ", "")
                    if alternate_indicator.replace(" ", "") != '':
                        #print(alternate_indicator)                         print(line2[13:27])
                        if not chainID+resSeq in alternate_indicator_keep_max:
                            alternate_indicator_keep_max[chainID+resSeq] = float(line2[54:60])
                        elif alternate_indicator_keep_max[chainID+resSeq]<float(line2[54:60]):
                            alternate_indicator_keep_max[chainID+resSeq] = float(line2[54:60])
                            coords.pop(); chains.pop(); residues.pop(); resSeqs.pop(); chainIDs.pop();
                        else:
                            continue
                    x, y, z = line2[30:38], line2[38:46], line2[46:54]
                    x, y, z = x.replace(" ", ""), y.replace(" ", ""), z.replace(" ", "")
                    x, y, z = float(x), float(y), float(z)
                    coords.append([x,y,z])
                    chains.append(chainID)
                    if chainID not in chainIDs:
                        chainIDs.append(chainID)
                    residues.append(resName)
                    resSeqs.append(resSeq)
    #print(alternate_indicator_keep_max)
    fileLines.close()
    chainFastas = {}
    for chain in chainIDs:
        chain_bool = np.array(chains) == chain
        chain_resSeq = np.array(resSeqs)[chain_bool].astype(np.int32) # PDB .pdb fasta
        #if len(chain_resSeq) == 0: continue
        chain_seq = np.array(residues)[chain_bool]
        chain_seq1= seq1((''.join(chain_seq)))
        chainFastas[chain] = chain_seq1
    return np.array(coords), chains,atomNames, residues, resSeqs, chainIDs, chainFastas



def s_score(distance,d0 = 5):
    return 1/(1+(distance/d0)**2)

def create_df(model_fs):
    has_models = []
    model_domains = []
    for model_f in model_fs:
        coordCA, chain,atomName, residue, resSeq, chainID,chainFasta = get_pdb_atom(model_f,'CA')
        model = model_f.split('/')[-1]
        has_models.append(model_f)
        model_domains.append(model)
    df_model_domain = pd.DataFrame()
    df_model_domain['model_domain'] = model_domains
    df_model_domain['has_model'] = has_models
    return df_model_domain

def read_backbone(df_model_domain):
    model_coords,  chains, residues, resSeqs, chainIDs,chainFastas = [],[],[],[],[],[]
    for i,row in df_model_domain.iterrows():
        # get chain-1 coord
        coordCA, chain,atomName, residue, resSeq, chainID,chainFasta = get_pdb_atom(row.has_model,'CA')
        coordN = get_pdb_atom(row.has_model,'N')[0]
        coordC = get_pdb_atom(row.has_model,'C')[0]
        coordCB = get_pdb_atom(row.has_model,'O')[0]
        model_coord = {'N':coordN,'CA':coordCA,'C':coordC,'O':coordCB}
        model_coords.append(model_coord)
        chainFastas.append(chainFasta)
        chains.append(chain);residues.append(residue);resSeqs.append(resSeq);chainIDs.append(chainID);
    df_model_domain['model_coord'] = model_coords
    df_model_domain['chain'] = chains
    df_model_domain['residue'] = residues
    df_model_domain['resSeq'] = resSeqs
    df_model_domain['chainID'] = chainIDs
    df_model_domain['chainFasta'] = chainFastas
    df_model_domain = df_model_domain.reset_index(drop=True)
    return df_model_domain

# Get distance map for the entire target (1 version at a time out of 5 versions)
# N_N, N_CA, N_C, N_O, CA_CA, CA_C, CA_O, C_C, C_O, O_O
def get_dist_map(row_coord):
    # coord<atom> is an array of coordinates for <atom> in every residue of the target
    # row_coord = {'N':coordN,'CA':coordCA,'C':coordC,'CB':coordCB}
    dist_maps = {}
    for i,key in enumerate(row_coord.keys()):
        # e.g. all N coords
        coord_i = row_coord[key]
        for j,key2 in enumerate(row_coord.keys()):
            if j<i: continue
            # e.g. all CA coords
            coord_j = row_coord[key2]
            # Generate distance matrix using scipy.spatial.distance_matrix()
            # e.g. distance between all coordNs and all coordCAs
#             print(f'\n\n\ncoord_i\n{coord_i}\n\ncoord_j{coord_j}\n\n\n')
            dist_mat = distance_matrix(coord_i, coord_j)
            #dist_mat_ij = dist_mat[np.ix_(chainIdxs[chaini], chainIdxs[chainj])]
            dist_maps[key+'_'+key2] = dist_mat
            #print(dist_mat.shape)
    return dist_maps

# '''
# Disregard:
# Should the threshold be set to 4 for a 81x81x81 cubic?

# -  -  -  -  -  -  -  -  -
# -  -  -  -  -  -  -  -  -
# -  -  -  -  -  -  -  -  -
# -  -  -  -  -  -  -  -  -
# -  -  -  -  CA -  -  -  -
# -  -  -  -  -  -  -  -  -
# -  -  -  -  -  -  -  -  -
# -  -  -  -  -  -  -  -  -
# -  -  -  -  -  -  -  -  -
# -40.5       0          40.5

# ...
# -2.5 - -1.5
# -1.5 - -0.5
# -0.5 - 0.5
# 0.5 - 1.5
# 1.5 - 2.5
# ...

# '''
def atom_neighbors_dist(row,threshold = 8):
    # row_coord = {'N':coordN,'CA':coordCA,'C':coordC,'O':coordO}
    neighbors = []
    # e.g. N_CA
    for i,key in enumerate(row.model_dist_map):
        # e.g. N, CA
        atom_type1,atom_type2 = key.split('_')
        # distance matrix of which N_CA distances are <= 8
#         print(f'\n{key}: row.model_dist_map[{key}]: {row.model_dist_map[key][0][0]}')
        '''
Distance matrix:
row.model_dist_map[N_N]: [[ 0.0          3.44973245  5.77563598 ... 53.63202742 55.55495949
  58.61856366]
 [ 3.44973245  0.0          3.36832317 ... 54.72813778 56.77635006
  59.87986602]
 [ 5.77563598  3.36832317  0.0         ... 55.37742184 57.53579195
  60.68972899]
 ...
 [53.63202742 54.72813778 55.37742184 ...  0.0          3.15002365
   6.17972321]
 [55.55495949 56.77635006 57.53579195 ...  3.15002365  0.0
   3.2680768 ]
 [58.61856366 59.87986602 60.68972899 ...  6.17972321  3.2680768
   0.0        ]]

Indices:
[[  0   0]
 [  0   1]
 [  0   2]
 ...
 [288 286]
 [288 287]
 [288 288]]
        '''
        # Get indices 
        neighbor_idx = np.argwhere((row.model_dist_map[key]<=threshold))
        '''
Indices where distance <= threshold:
print(neighbor_idx)
[[  0   0]
 [  0   1]
 [  0   2]
 ...
 [288 286]
 [288 287]
 [288 288]]
        '''
        
        
        '''
        Shouldn't identical residue indices be included if the atoms are different?
        Because you don't want to include duplicate atoms in the same residue, but can include other atoms in the same residue.
        (i<j if atom_type1 == atom_type2 else i<=j)
        
        res 1 CA N C O
        res 2 CA N C O
        '''
        neighbor_idx = np.array([[i,j] for i,j in neighbor_idx if (i<j if atom_type1 == atom_type2 else i<=j)])
#         print(f'{key}: {neighbor_idx[0][0]} {neighbor_idx[0][1]}\n')
        '''
Indices where distance <= threshold and remove identical and duplicate atoms:
print(neighbor_idx)
[[  0   1]
 [  0   2]
 [  1   2]
 ...
 [286 287]
 [286 288]
 [287 288]]
        '''
        # Get all distances at all row indices, column indices in neighbor_idx
        # neighbor_idx are the indices that store the distances in neighbor_dist
        neighbor_dist = row.model_dist_map[key][neighbor_idx[:,0],neighbor_idx[:,1]]
        '''
neighbor_dist
[3.44973245 5.77563598 3.36832317 ... 3.15002365 6.17972321 3.2680768]
        '''
        for j,num1_num2 in enumerate(neighbor_idx):
            # num1, num2 is essentially the row, column index (residue index)
            num1,num2 = num1_num2
            # e.g. N_0
            res1 = f'{atom_type1}_{num1}'
            # e.g. CA_1
            res2 = f'{atom_type2}_{num2}'
            # e.g. [N_0, CA_1, 3.450]
            neighbors.append([res1,res2,neighbor_dist[j]])
#         print(neighbors)
#         exit()
    '''
print(neighbors)
[['N_0', 'CA_1', 4.727414303824026], ['N_0', 'CA_2', 6.996543503759555], ['N_1', 'CA_2', 4.711009339833663] ... ['N_286', 'CA_287', 4.554294896029462], ['N_286', 'CA_288', 7.566301936877756], ['N_287', 'CA_288', 4.704681073144065]]
    '''
    return np.array(neighbors)

def parser_local_prediction(f_):
    model = f_.split('/')[-2].replace('.2','')
    local_scores = []
    for line in open(f_,'r'):
        terms = line.rstrip().split()
        if len(terms)<1: continue
        if terms[0] in ['PFRMAT','AUTHOR','END']: continue
        if model in terms[0]: terms = terms[2:]
        local_scores+=terms # same thing as .extend()
    return np.array(local_scores).astype(float)

def one_hot_coding(seqCat,atom):
#     print(f'seqCat {seqCat}')
#     print(f'atom {atom}')
    map_idx = {'M': 0, 'K': 1, 'A': 2, 'D': 3, 'W': 4, 'E': 5, 'L': 6, 'T': 7, 'F': 8,
               'G': 9, 'R': 10, 'V': 11, 'I': 12, 'C': 13, 'Y': 14, 'S': 15, 'H': 16, 
               'N': 17, 'P': 18, 'Q': 19, 'X': 20, 'U': 20}
    map_idx_atom_type = {'N':0,'CA':1,'C':2,'O':3}
    hot_code = np.zeros((2,len(seqCat)))
    for i,aci in enumerate(seqCat):
        if aci in map_idx.keys():
            idx = map_idx[aci]
        else:
            idx = 20
        hot_code[0,i] = (idx+1)/10
    hot_code[1,:] = (map_idx_atom_type[atom]+1)/10
    # hot_code.T is a two-dimensional where each residue has [encoded resCode, encoded atomCode]
    return hot_code.T

def sinusoidal_positional_encoding(length):
    '''
    input: n is an integer
    output: a 2D list (n * 4)
    '''
    spe = []
    for i in range(1, length + 1):
        tmp = [math.sin(i), math.cos(i), math.sin(i/100), math.cos(i/100)]
        tmp = [round(x, 6) for x in tmp]
        spe.append(tmp)
    return np.array(spe)

def generate_atom_level_traning_samples(row, resolution, cubic_size, positional_encoding_max):
    training_samples = [] # model_doamin, residue, resNum, atom, feature 11^3, target 11^3
    # map index
    CB_i = 0
    CBi2i = {} # CBi is only used in CB coord distance map
    for i, residue in enumerate(row.residue):
        if residue!='GLY':
            CBi2i[CB_i] = i
            CB_i += 1
    CB_i = 0
    # 10 channels of features
    N_feat = 10
    # Counter for number of neighboring atoms, labels, & number of atoms centered in the cubic
    counters = {'numb_neighbor_atom':[],'label_out_cubic':0,'atom':0}
    for i, residue in enumerate(row.residue):
        #print(row.model_domain,i,residue)
        for atom in ['N','CA','C','O']:
            # coord_feats is an array containing an array for every coordinate with features. The first element in the inner array is the array of coordinates (e.g. 40,40,40 for middle), and the second element in the inner array is an array of each feature ([onehot, mass law score, sinusoidal encoding, esm stat])
            coord_feats = []
            if residue == 'GLY' and atom == 'CB': continue
            # Counter; Atom
            # 81 x 81 x 81 array filled with zeros
            counter = np.zeros((int(cubic_size),int(cubic_size),int(cubic_size)))
            # [40,40,40]
            atom_in_cubic = [int(cubic_size/2),int(cubic_size/2),int(cubic_size/2)]
            # Set middle of cubic to 1
            # counter[40,40,40] = 1
            counter[atom_in_cubic[0],atom_in_cubic[1],atom_in_cubic[2]] = 1
            counters['atom']+=1
            if atom == 'CB':
                # update to full idx
                atom_idx = f'{atom}_{CB_i}'
                atom_coord = row.model_coord[atom][CB_i,:]
                atom_feat = np.concatenate([row.onehot[atom][CBi2i[CB_i]],row.mass_law[CBi2i[CB_i]],\
                                            positional_encoding_max[CBi2i[CB_i]],row.esm_stat[:,:1][CBi2i[CB_i]]])
            else:
                # e.g. N_0 (residue index 0)
                atom_idx = f'{atom}_{i}'
                # AlphaFold coord for atom at residue index
                atom_coord = row.model_coord[atom][i,:]
                # For atom's other features, combine one hot, mass_law score, sinusoidal positional encoding, and esm stat in one array
#                 print(f'One-hot encoding\n{row.onehot[atom][i]}\n\nMass-Law score\n{row.mass_law[i]}\n\nSinusoidal encoding\n{positional_encoding_max[i]}\n\nESM stat\n{row.esm_stat[:,:1][i]}\n')
                atom_feat = np.concatenate([row.onehot[atom][i],row.mass_law[i],\
                                            positional_encoding_max[i],row.esm_stat[:,:1][i]])
            # Append the atom features to empty coord features array
            coord_feats.append([atom_in_cubic,atom_feat])
            # Get neighbor distance data for version where the second atom (e.g. N_0) == current atom: get the first atom, do the same thing but where the first atom == current atom, get the second atom
            # Essentially get a list of the other atoms that are neighbors of the current atom
            # np.concatentate([first atoms],[second atoms])
            neighbors = np.concatenate([row.atom_neighbor_dist[row.atom_neighbor_dist[:,1] == atom_idx][:,0],
                                        row.atom_neighbor_dist[row.atom_neighbor_dist[:,0] == atom_idx][:,1]])
            # Neighbors
            for neighbor in neighbors:
                # e.g. CA, 1
                n_atom,n_idx = neighbor.split('_') # i or CB_i
                n_idx = int(n_idx)
                # Get AlphaFold coord for neighboring atom at its residue index ([x,y,z])
                n_atom_coord = row.model_coord[n_atom][n_idx,:] # no need to map
                # n_atom_coord & atom_coord are NumPy arrays?
                # difference in coords / resolution (0.1), position in regards to center
                '''
                Change to round to nearest rather than always rounding down?
                Features would always be closer to the center than they would be, e.g. -2.29 would round down to -22, instead of rounding to -23
                
                0.29 --> 2.9 --> 2.0
                0.09 --> 0.9 --> 1
                -0.09 --> -0.9 --> -1
                '''
                # Round to nearest int instead of integer division
                n_in_cubic = np.array(np.round((n_atom_coord-atom_coord)/resolution))+\
                                np.array([int(cubic_size/2),int(cubic_size/2),int(cubic_size/2)])
                # If out of range (< 0 or >= 81), ignore. Out of range atoms were gotten as the distance matrix threshold was set to 8
                if np.any(n_in_cubic<0) or np.any(n_in_cubic>=cubic_size):
                    continue
                n_in_cubic = n_in_cubic.astype(int)
                # Add a one to the position of neighbor atom in cubic
                counter[n_in_cubic[0],n_in_cubic[1],n_in_cubic[2]]+=1
                if n_atom == 'CB':
                    # update to full idx
                    n_feat =  np.concatenate([row.onehot[n_atom][CBi2i[n_idx]],row.mass_law[CBi2i[n_idx]],\
                                            positional_encoding_max[CBi2i[n_idx]],row.esm_stat[:,:1][CBi2i[n_idx]]])
                else:
                    # For neighboring atom's other features, combine one hot, mass_law score, sinusoidal positional encoding, and esm stat in one array
                    n_feat = np.concatenate([row.onehot[atom][n_idx],row.mass_law[n_idx],\
                                            positional_encoding_max[n_idx],row.esm_stat[:,:1][n_idx]])
                # Add neighbor's feature to the features at the neighbor's coord
                coord_feats.append([n_in_cubic,n_feat])
            # Number of neighboring atoms
            counters['numb_neighbor_atom'].append(len(coord_feats)-1)
            # Initialize empty features 4D array (10 channels of features)
            atom_featS = np.zeros((N_feat,cubic_size,cubic_size,cubic_size))
            # Counter is the 3D numpy array with 1s at atom/neighbor positions & 0s everywhere else
            counter_tmp = counter.copy()
            '''
            Does this line do anything?
            '''
            counter_tmp[counter_tmp==0]=0
            # Set 1st channel of features equal to the 81x81x81 grid of 1s & 0s for the atom
            atom_featS[0] = counter_tmp
            # divider
            divider = counter.copy()
            # Set all 0s to 1s
            divider[divider<1] = 1
            # For each feature array ([coords in cubic, features array])
            for pair in coord_feats:
                # Coords (n_in_cubic)
                to_x,to_y,to_z = pair[0]
                # Feature array [onehot, mass_law score, sinusoidal positional encoding, and esm stat]
                feat_ = pair[1]
                '''
                atoms_feats is only 4D?
                feat_ [onehot, onehot, mass-law, mass-law, sinusoidal, sinusoidal, sinusoidal, sinusoidal, esm]
                '''
                atom_featS[1:3][:,to_x,to_y,to_z] += feat_[:2]
                atom_featS[3:5][:,to_x,to_y,to_z] += feat_[2:4]
                atom_featS[5:9][:,to_x,to_y,to_z] += feat_[4:8]
                atom_featS[9][to_x,to_y,to_z] += feat_[8]
            # Normalize features?
            atom_featS = atom_featS/divider
            # Make sparse tensor
            atom_featS = torch.from_numpy(atom_featS).to_sparse()
            # Make training data row
            training_sample = [row.model_domain, i,residue, row.resSeq[i], atom, atom_featS,atom_coord]
            # Add row to list
            training_samples.append(training_sample)
        if residue != 'GLY':
            CB_i += 1
    # Create dataframe
    df_training_atoms = pd.DataFrame(training_samples, columns = ['model_domain','idx', 'residue', 'resSeq', 'atom', 'features','atom_coord'])
    return counters, df_training_atoms

def main():
#     allTargets = ['T1129s2', 'T1133', 'T1134s1', 'T1134s2', 'T1137s1', 'T1137s2', 'T1137s3', 'T1137s4', 'T1137s5', 'T1137s6', 'T1137s7', 'T1137s8', 'T1137s9', 'T1145', 'T1151s2', 'T1152', 'T1159', 'T1170', 'T1176', 'T1185s1', 'T1185s2', 'T1185s4', 'T1187', 'T1188']
#     target = sys.argv[1]
#     for target in allTargets:
#         if len(sys.argv) > 1:
    target = sys.argv[1]
#             print('Only doing one target', flush=True)
#         print(f'{target} start', flush=True)
#         af_models = glob.glob(f'{unpress_dir}/{target}/af2*')
#         af_models_standards = []
#         if len(af_models) == 10:
#             for af_model in af_models:
#                 if 'af2-standard' in af_model:
#                     af_models_standards.append(af_model)
#             af_models = af_models_standards
#         #print(af_models)
#         assert len(af_models) == 5

    df_model_domain = create_df([target])#create_df(af_models)
        #df_model_domain = df_model_domain[df_model_domain.model_domain.str.contains('multimer') == False] # only ts
    df_model_domain = read_backbone(df_model_domain)
    df_model_domain.to_pickle(f'{temp_dir}df.pkl')
#     if not os.path.exists(f'{featrue_dir}{target}_esm.pkl') or True:
    # esm
    '''
    What is esm???
    '''
    os.system(f'python {script_dir}/util/run_esm.py {temp_dir}df.pkl')
    df_model_domain = pd.read_pickle(f'{temp_dir}df_esm.pkl')
    # dist map
    model_dist_maps = []
    # For each of the 5 (AlphaFold model) versions
    for i,row in df_model_domain.iterrows():
        model_dist_maps.append(get_dist_map(row.model_coord))
    df_model_domain['model_dist_map'] = model_dist_maps
    # find neighbor
    atom_neighbor_dists = []
    for i,row in df_model_domain.iterrows():
        atom_neighbor_dists.append(atom_neighbors_dist(row))
    df_model_domain['atom_neighbor_dist'] = atom_neighbor_dists

    # rad mass law
    mass_law_model = target.split('/')[-1]
    mass_law = []
    for i,row in df_model_domain.iterrows():
        model = row.model_domain
        mass_f = f'{mass_law_casp14_job_dir}/{mass_law_model}.2/{mass_law_model}_MASS-CASP14.txt'
        mass_local_score = s_score(parser_local_prediction(mass_f))
        law_f = f'{mass_law_casp14_job_dir}/{mass_law_model}.2/{mass_law_model}_LAW-CASP14.txt'
        law_local_score = s_score(parser_local_prediction(law_f))
        mass_law_local_score = np.array([mass_local_score,law_local_score]).T
        mass_law.append(mass_law_local_score)
    df_model_domain['mass_law'] = mass_law    

    # one hot coding
    onehot = []
    for i,row in df_model_domain.iterrows():
        one_hot_codings = {}
        one_hot_codings['N'] = one_hot_coding(row.chainFasta[' '],'N')
        one_hot_codings['CA'] =one_hot_coding(row.chainFasta[' '],'CA')
        one_hot_codings['C'] =one_hot_coding(row.chainFasta[' '],'C')
        one_hot_codings['O'] =one_hot_coding(row.chainFasta[' '],'O')
        onehot.append(one_hot_codings)
    df_model_domain['onehot'] = onehot
    df_model_domain.to_pickle(f'{temp_dir}df_esm.pkl')
#     else:
# #         # Ross Test
#         print(f'{featrue_dir}{target}_esm.pkl already exists', flush=True)
# #         exit()
#         df_model_domain = pd.read_pickle(f'{featrue_dir}{target}_esm.pkl')
    # save to pkl 
    resolution = 0.1
    cubic_size = 40*2+1
#     save_dir = featrue_dir
    positional_encoding_max = sinusoidal_positional_encoding(99999)
    counters_numb_neighbor_atom,counters_label_out_cubic,counters_atom = [],0,0
#     for i,row in df_model_domain.iterrows():
#         if os.path.exists(f'{save_dir}{row.model_domain}.pkl'):continue
    counters, df_training_atoms = generate_atom_level_traning_samples(df_model_domain.iloc[0], resolution, cubic_size, positional_encoding_max)
    df_training_atoms.to_pickle(f'{temp_dir}features.pkl')
    print(f'{temp_dir}features.pkl')
#         print(f'{save_dir}{row.model_domain}.pkl')
#     print(f'{target} done', flush=True)
#     if len(sys.argv) > 1:
#         print('Exiting', flush=True)
#         exit()

if __name__ == '__main__':
    main()
