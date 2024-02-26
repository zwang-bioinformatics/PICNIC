import os,sys,time
import pandas as pd
import numpy as np
import scipy.stats
import torch
import esm
# load esm
model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
batch_converter = alphabet.get_batch_converter()

def df_esm(df):
    esmseq,esmstat,esmnorm,esmmean = [],[],[],[]
    for i, row in df.iterrows():
        data = [(row.model_domain,row.chainFasta[' '])]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[34])
        token_embeddings = results["representations"][34]
        pred = token_embeddings[0].numpy()
        #ptp_ = np.ptp(pred,axis=1)
        mean_ = np.mean(pred,axis=1)
        median_ = np.median(pred,axis=1)
        std_ = np.std(pred,axis=1)
        esm_stat = np.array([mean_,median_,std_]).T
        esmseq.append(pred)
        esmmean.append(mean_)
        esmstat.append(esm_stat)
        esmnorm.append(scipy.special.expit((scipy.stats.zscore(pred,axis=1))))
    #df['esm'] = esmseq
    df['esm_stat'] = esmstat
    #df['esm_norm'] = esmnorm
    #df['esm_mean'] = esmmean
    return df

df_ = pd.read_pickle(sys.argv[1])
df_w_esm  = df_esm(df_)
df_w_esm.to_pickle(sys.argv[1].replace('.pkl','_esm.pkl'))
