import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import myvariant

def mutation_ratio(id1, id2, df_sum, res):

    tmp1 = df_sum.loc[(df_sum['SYMBOL'] == id1)]
    tmp2 = df_sum.loc[(df_sum['SYMBOL'] == id2)]

    tmp1.reset_index(inplace=True)
    tmp2.reset_index(inplace=True)

    tmp1 = tmp1.drop(columns=['SYMBOL', 'index'])
    tmp2 = tmp2.drop(columns=['SYMBOL', 'index'])

    d = id1 + '-' + id2

    tmp_add = tmp1.add(tmp2, fill_value=0)
    tmp_add.rename(index={0:d}, inplace=True)
    
    if not tmp_add.empty:
        tmp_ratio = tmp_add.div(tmp_add['MAX'], axis=0)
        tmp_ratio = 1 - tmp_ratio # mutation ratio to edge affection score
        tmp_ratio.drop(columns=['MAX'], inplace=True)
        res = pd.concat([tmp_ratio, res])

    else:
        tmp_ratio = tmp_add.copy()
        tmp_ratio.loc[d] = 1
        res = pd.concat([tmp_ratio, res])

    return res

def obtain_edges_scores(G, data):
    
    samples_df = data.drop(columns=['SYMBOL'])
    samples    = samples_df.columns

    # For PPIs
    data[samples] = data[samples].astype('float')
    ppi_sum = data.groupby(['SYMBOL'])[samples].sum()
    ppi_sum_max = ppi_sum.assign(MAX=ppi_sum.max(axis=1).values)    
    ppi_sum_max.reset_index(inplace=True)

    edgelist = G.edges(data=True)
    results_ppi = pd.DataFrame()
    for prot1, prot2, attributes in edgelist:
        
        if prot1 != prot2: # TODO: self-loops are not contemplated at the moment
            id1 = str(prot1)
            id2 = str(prot2)
            results_ppi = mutation_ratio(id1, id2, ppi_sum_max, results_ppi)
        else:
            pass

    return results_ppi
