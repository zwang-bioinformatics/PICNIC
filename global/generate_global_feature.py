import glob
import os,sys
import pandas as pd 
from Bio.SeqUtils import seq3,seq1
import numpy as np
import math
import torch

# server_dir = './'
# unpress_dir = f'{server_dir}/data-casp15/unpress_TS_model/'
# featrue_dir = f'{server_dir}/data-casp15/feature_global_TS_model_0-6_res/'
# mass_law_casp14_dir = '../MASS-CASP14_LAW-CASP14_Server/'
# mass_law_casp14_job_dir = f'{mass_law_casp14_dir}data-casp15-af2/'
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = f'{script_dir}/.var/'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

def get_pdb_atom(file_pdb,atom_):
    """
    Read a certain atom from pdb file
    return coords, chains,atomNames, residues, resSeqs, chainIDs, chainFastas
    """
    coords, chains, atomNames, residues, resSeqs, chainIDs, af_QA_scores = [],[],[],[],[],[],[]
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
                    af_QA_score = float(line2[61:66].replace(' ','')) / 100
                    coords.append([x,y,z])
                    chains.append(chainID)
                    if chainID not in chainIDs:
                        chainIDs.append(chainID)
                    residues.append(resName)
                    resSeqs.append(resSeq)
                    af_QA_scores.append(af_QA_score)
    #print(alternate_indicator_keep_max)
    fileLines.close()
    chainFastas = {}
    for chain in chainIDs:
        chain_bool = np.array(chains) == chain
        # resSeqs can sometimes have letters e.g. '32A'
#         chain_resSeq = np.array(resSeqs)[chain_bool].astype(np.int32) # PDB .pdb fasta
        #if len(chain_resSeq) == 0: continue
        chain_seq = np.array(residues)[chain_bool]
        chain_seq1= seq1((''.join(chain_seq)))
        chainFastas[chain] = chain_seq1
    return np.array(coords), chains,atomNames, residues, resSeqs, chainIDs, chainFastas, af_QA_scores


def s_score(distance,d0 = 5):
    return 1/(1+(distance/d0)**2)

def create_df(model_fs):
    has_models = []
    model_domains = []
    for model_f in model_fs:
        coordCA, chain,atomName, residue, resSeq, chainID,chainFasta, af_QA_score = get_pdb_atom(model_f,'CA')
        model = model_f.split('/')[-1]
        has_models.append(model_f)
        model_domains.append(model)
    df_model_domain = pd.DataFrame()
    df_model_domain['model_domain'] = model_domains
    df_model_domain['has_model'] = has_models
    return df_model_domain

def read_backbone(df_model_domain):
    model_coords,  chains, residues, resSeqs, chainIDs,chainFastas, af_QA_scores = [],[],[],[],[],[],[]
    for i,row in df_model_domain.iterrows():
        # get chain-1 coord
        coordCA, chain,atomName, residue, resSeq, chainID,chainFasta, af_QA_score = get_pdb_atom(row.has_model,'CA')
        coordN = get_pdb_atom(row.has_model,'N')[0]
        coordC = get_pdb_atom(row.has_model,'C')[0]
        coordCB = get_pdb_atom(row.has_model,'O')[0]
        model_coord = {'N':coordN,'CA':coordCA,'C':coordC,'O':coordCB}
        model_coords.append(model_coord)
        chainFastas.append(chainFasta)
        chains.append(chain);residues.append(residue);resSeqs.append(resSeq);chainIDs.append(chainID);af_QA_scores.append(af_QA_score);
    df_model_domain['model_coord'] = model_coords
    df_model_domain['chain'] = chains
    df_model_domain['residue'] = residues
    df_model_domain['resSeq'] = resSeqs
    df_model_domain['chainID'] = chainIDs
    df_model_domain['chainFasta'] = chainFastas
    df_model_domain['af_QA_score'] = af_QA_scores
    df_model_domain = df_model_domain.reset_index(drop=True)
    return df_model_domain

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

def generate_global_training_samples(row, resolution, positional_encoding_max, all_atoms):
    
    atom = 'CA'
    training_samples = [] # model_doamin, residue, resNum, atom, feature 11^3, target 11^3
    # Can refine atoms beyond the bounds of original protein
    cuboid_search_radius = 4 # 2 / .2 = 10, 4 Å away max include (previously was 4 / .2 = 20)
    # Side lengths of cuboid (not necessarily equal)
#     print(row.model_coord[atom])
    min_x, min_y, min_z = round(min(row.model_coord[atom][:,0])/resolution), round(min(row.model_coord[atom][:,1])/resolution), round(min(row.model_coord[atom][:,2])/resolution)
    x_length, y_length, z_length = round(max(row.model_coord[atom][:,0])/resolution) - min_x + 1 + 2*cuboid_search_radius,round(max(row.model_coord[atom][:,1])/resolution) - min_y + 1 + 2*cuboid_search_radius,round(max(row.model_coord[atom][:,2])/resolution) - min_z + 1 + 2*cuboid_search_radius
    xyz_cuboid_mins = [min_x,min_y,min_z] # [round(num/resolution) for num in [min_x,min_y,min_z]]
#     print(x_length,y_length,z_length)
    
    N_feat = 10
    counters = {'numb_neighbor_ca':[],'numb_neighbor_atom':[],'label_out_cubic':0,'atom':0}
    
#     print('3')
    coord_feats = []
    far_indices = []
    atom_coords = {all_atoms[0]:[],all_atoms[1]:[],all_atoms[2]:[],all_atoms[3]:[]}#, atom_true_coords = [],[]
    cuboid = np.zeros((x_length,y_length,z_length))
#     true_cuboid = np.zeros((x_length,y_length,z_length))#, dtype=int8)
#     print('lengths',x_length,y_length,z_length)
#     print('4')
    for i, residue in enumerate(row.residue):
#         atom_true_coords.append(atom_true_coord)
        atom_idx = f'{atom}_{i}'
        atom_coord = row.model_coord[atom][i,:]
        '''
        Normalize in cuboid: Scale with resolution and subtract cuboid mins to align lowest with index 0.
        Add search radius to allow for buffer space when refining.
        '''
        atom_coord_in_cuboid = (np.array(np.round(atom_coord/resolution))-\
                            np.array(xyz_cuboid_mins) + np.array([cuboid_search_radius,cuboid_search_radius,cuboid_search_radius])).astype(int)
        cuboid[atom_coord_in_cuboid[0],atom_coord_in_cuboid[1],atom_coord_in_cuboid[2]] += 1
        atom_feat = np.concatenate([row.onehot[atom][i],row.mass_law[i],\
                                    positional_encoding_max[i],row.esm_stat[:,:1][i]])
        coord_feats.append([atom_coord_in_cuboid,atom_feat])
        for atom_to_move_residue in atom_coords:
            atom_coords[atom_to_move_residue].append(row.model_coord[atom_to_move_residue][i,:])
    # Put coord_feats in cuboid
    global_feats = np.zeros((N_feat,x_length,y_length,z_length))
    counter_tmp = cuboid.copy()
#         counter_tmp[counter_tmp==0]=0 # used -1 before
    global_feats[0] = counter_tmp
    # divider
    divider = cuboid.copy()
    divider[divider<1] = 1
    '''
    1 1 1 1 1 1 2 1 1
    1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1
    etc.
    '''
    # First channel is the main feature and only label, other 9 features are used as extra data
    for pair in coord_feats:
        to_x,to_y,to_z = pair[0]
        feat_ = pair[1]
        global_feats[1:3][:,to_x,to_y,to_z] += feat_[:2]
        global_feats[3:5][:,to_x,to_y,to_z] += feat_[2:4]
        global_feats[5:9][:,to_x,to_y,to_z] += feat_[4:8]
        global_feats[9][to_x,to_y,to_z] += feat_[8]
    # Essentially have 10 copies of divider "broadcasted" along first dimension, allowing for division
    # Getting rid of values higher than 1?
    global_feats = global_feats/divider
    # Dense --> Sparse
    global_feats = torch.from_numpy(global_feats).to_sparse()
#     true_cuboid = torch.from_numpy(true_cuboid).to_sparse()
    
    # Take out outlier far residues
#     print(f'{row.model_domain} far_indices {far_indices}',flush=True)
#     if len(far_indices) >= 0.5*global_feats[0]._nnz():
#         print('bad alignment',flush=True)
#         return counters, None, 1

    # Doesn't do anything if native structure is not known
    far_indices = []
    indices = list(range(len(row.residue)))
    indices = [indices[i] for i in range(len(indices)) if i not in far_indices]
    residues = [row.residue[i] for i in range(len(row.residue)) if i not in far_indices]
    resSeqs = [row.resSeq[i] for i in range(len(row.resSeq)) if i not in far_indices]
#     print(f'{row.model_domain} global_feats[0]._nnz() {global_feats[0]._nnz()}',flush=True)
#     print(len(indices) , len(residues) , len(resSeqs) , global_feats[0]._nnz())
    # nnz() can be less than the others if there are two residues in same spot
#     print(len(atom_coords[all_atoms[0]]) , len(indices) , len(residues) , len(resSeqs) , global_feats[0]._nnz())
    assert len(atom_coords[all_atoms[0]]) == len(atom_coords[all_atoms[1]]) == len(atom_coords[all_atoms[2]]) == len(atom_coords[all_atoms[3]]) == len(indices) == len(residues) == len(resSeqs) and len(resSeqs) == global_feats[0]._nnz()
#     model_domains = [row.model_domain] * len(indices)
    training_sample = [row.model_domain, indices,residues, resSeqs, all_atoms, global_feats,atom_coords]
    training_samples.append(training_sample)
    # Create dataframe
    df_training_atoms = pd.DataFrame(training_samples, columns = ['model_domain','idx', 'residue', 'resSeq', 'all_atom', 'features','atom_coord'])
    return counters, df_training_atoms, 0

def main():
    # Validatation targets
#     allTargets = ['T1067','T1073','T1074','T1082','T1099']
#     allTargets = ['T1104', 'T1106s1', 'T1106s2', 'T1109', 'T1110', 'T1112', 'T1113', 'T1114s1', 'T1114s2', 'T1114s3', 'T1115', 'T1119', 'T1120', 'T1121', 'T1122', 'T1123', 'T1124', 'T1125', 'T1127', 'T1129s2', 'T1130', 'T1131', 'T1132', 'T1133', 'T1134s1', 'T1134s2', 'T1137s1', 'T1137s2', 'T1137s3', 'T1137s4', 'T1137s5', 'T1137s6', 'T1137s7', 'T1137s8', 'T1137s9', 'T1139', 'T1145', 'T1146', 'T1147', 'T1150', 'T1151s2', 'T1152', 'T1153', 'T1154', 'T1155', 'T1157s1', 'T1157s2', 'T1158', 'T1159', 'T1160', 'T1161', 'T1162', 'T1163', 'T1165', 'T1169', 'T1170', 'T1173', 'T1174', 'T1175', 'T1176', 'T1177', 'T1178', 'T1179', 'T1180', 'T1181', 'T1182', 'T1183', 'T1184', 'T1185s1', 'T1185s2', 'T1185s4', 'T1187', 'T1188', 'T1194', 'T1195', 'T1196', 'T1197']
#     allTargets = set(allTargets)
#     target = sys.argv[1]
#     for target in allTargets:
#     if len(glob.glob(f'./data-casp15/feature_global_TS_model_0-6_res/af2-standard_{target}_*.pkl')) >= 5:
#         print(f'./data-casp15/feature_global_TS_model_0-6_res/af2-standard_{target}_*.pkl already exists')
#         continue
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <pdb_path>"
    target_f = sys.argv[1]
#         print('Only doing one target', flush=True)
#     print(f'{target} start', flush=True)
#     af_models = glob.glob(f'{unpress_dir}/{target}/af2*')
#         af_models = glob.glob(f'{unpress_dir}/{target}/*')
#         af_models = [i for i in af_models if not '.tmscore' in i]
#     af_models_standards = []
#     if len(af_models) == 10:
#         for af_model in af_models:
#             if 'af2-standard' in af_model:
#                 af_models_standards.append(af_model)
#         af_models = af_models_standards
    #print(af_models)
#     assert len(af_models) == 5

    df_model_domain = create_df([target_f])#create_df(af_models)
    #df_model_domain = df_model_domain[df_model_domain.model_domain.str.contains('multimer') == False] # only ts
    df_model_domain = read_backbone(df_model_domain)
    df_model_domain.to_pickle(f'{temp_dir}df.pkl')
#     if not os.path.exists(f'{featrue_dir}{target}_esm.pkl') or True:
        # esm
    os.system(f'python {script_dir}/util/run_esm.py {temp_dir}df.pkl')
    df_model_domain = pd.read_pickle(f'{temp_dir}df_esm.pkl')
    # dist map
#             model_dist_maps = []
#             # For each of the 5 (AlphaFold model) versions
#             for i,row in df_model_domain.iterrows():
#                 model_dist_maps.append(get_dist_map(row.model_coord))
#             df_model_domain['model_dist_map'] = model_dist_maps
#             # find neighbor
#             atom_neighbor_dists = []
#             for i,row in df_model_domain.iterrows():
#                 atom_neighbor_dists.append(atom_neighbors_dist(row))
#             df_model_domain['atom_neighbor_dist'] = atom_neighbor_dists

    # AF QA score for now
    mass_law = []
    for i,row in df_model_domain.iterrows():
        af_QA_score = np.array([row.af_QA_score,row.af_QA_score]).T
        mass_law.append(af_QA_score)
#                 model = row.model_domain
#                 mass_f = f'{mass_law_casp14_job_dir}/{model}.2/{model}_MASS-CASP14.txt'
#                 mass_local_score = s_score(parser_local_prediction(mass_f))
#                 law_f = f'{mass_law_casp14_job_dir}/{model}.2/{model}_LAW-CASP14.txt'
#                 law_local_score = s_score(parser_local_prediction(law_f))
#                 mass_law_local_score = np.array([mass_local_score,law_local_score]).T
#                 mass_law.append(mass_law_local_score)
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
    # save to pkl 
    resolution = 0.6
    positional_encoding_max = sinusoidal_positional_encoding(99999)
    counters_numb_neighbor_atom,counters_label_out_cubic,counters_atom = [],0,0
    
    # 1 row
    counters, df_training_atoms, exit_code = generate_global_training_samples(df_model_domain.iloc[0], resolution, positional_encoding_max, ['N','CA','C','O'])
    if exit_code == 1:
#         print('df training atoms null',flush=True)
        exit(1)
    df_training_atoms.to_pickle(f'{temp_dir}features.pkl')
#     print(f'generated features at {temp_dir}features.pkl')
    print(f'{temp_dir}features.pkl')

if __name__ == '__main__':
    main()