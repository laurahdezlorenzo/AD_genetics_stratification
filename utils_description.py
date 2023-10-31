import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib_venn as mvenn
from matplotlib import pyplot as plt
import gseapy as gp
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_diagnosis_info():

    adnimerge    = pd.read_csv('data/ADNI/ADNIMERGE.csv', index_col='PTID', low_memory=False)
    adnimerge_bl = adnimerge.loc[adnimerge['VISCODE'] == 'bl']

    # Last available visits
    sample_dfs = []
    for sample in set(adnimerge.index.to_list()):
        
        tmp = adnimerge.loc[sample].copy()
        tmp['Month'] = tmp['Month'].astype(float)
        if type(tmp) == pd.DataFrame:
            tmp_dem = tmp.loc[tmp['DX'] == 'Dementia']
            tmp_mci = tmp.loc[tmp['DX'] == 'MCI']
            
            max_month     = tmp.dropna(subset=['DX'])['Month'].max()
            month_dem = tmp_dem.dropna(subset=['DX'])['Month'].min()
            month_mci = tmp_mci.dropna(subset=['DX'])['Month'].min()
            
            tmp['Month_Dementia'] = month_dem
            tmp['Month_MCI'] = month_mci
                        
            sample_dfs.append(tmp.loc[tmp['Month'] == max_month])
        
    diagnose_lt = pd.concat(sample_dfs)
    diagnose_lt = diagnose_lt[~diagnose_lt.index.duplicated(keep='first')]

    result = diagnose_lt
    
    result.to_csv('data/ADNI/ADNIMERGE_processed.csv')

    return result

def obtain_cluster_graphs_edges_scores(edges_data, clusters_data, original_G):

    print('Original', original_G)
    random_pos = nx.random_layout(original_G, seed=42)
    pos = nx.spring_layout(original_G, pos=random_pos)

    # Add unit weights to original network
    nx.set_edge_attributes(original_G, values = 1, name = 'weight')

    edges_data = edges_data.loc[clusters_data.index] # select clusters' patients
    
    for i in clusters_data.unique():

        tmp_edges_scores = edges_data.loc[clusters_data == i]

        mean_edges_cluster = pd.DataFrame(tmp_edges_scores.mean().rename('weight')).reset_index()
        mean_edges_cluster[['intA', 'intB']] = mean_edges_cluster['index'].str.split('-', 1, expand=True)
        mean_edges_cluster = mean_edges_cluster[['intA', 'intB', 'weight']]
        # print(mean_edges_cluster)

        cluster_G = nx.from_pandas_edgelist(mean_edges_cluster, 'intA', 'intB', edge_attr='weight')
        
        print('Cluster', i, cluster_G)

        nx.write_gexf(cluster_G, f'results/cluster{i}.gexf')
        nx.write_edgelist(cluster_G, f'results/cluster{i}.edgelist')
    
def plot_weighted_network(G, pos, title):
   
    nodelist = G.nodes()
    
    edge_weights = nx.get_edge_attributes(G,'weight')
    G.remove_edges_from((e for e, w in edge_weights.items() if w < 0.1))

    # Edge widths and colors
    widths = nx.get_edge_attributes(G, 'weight')
    edge_colors = [G[u][v]['weight'] for u, v in G.edges]
    edge_weights = list(widths.values())
    edge_weights = [e * 5 for e in edge_weights]

    plt.figure(figsize=(20, 20))

    nx.draw_networkx_nodes(G, pos,
                            nodelist=nodelist,
                            node_size=2000,
                            node_color='lightblue')

    map = nx.draw_networkx_edges(G, pos,
                            edgelist = widths.keys(),
                            width = edge_weights,
                            edge_color = edge_colors,
                            edge_cmap = plt.cm.gray_r,
                            edge_vmax=1.,
                            edge_vmin=0.)

    nx.draw_networkx_labels(G, pos=pos,
                            font_size=14,
                            labels=dict(zip(nodelist,nodelist)),
                            font_color='black')

    plt.box(False)
    plt.tight_layout()
    plt.show()

def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)), 3)

def weighted_jaccard(g, h):
    edges = set(g.edges()).union(h.edges())
    mins, maxs = 0, 0

    for edge in edges:
        weight_g = g.get_edge_data(*edge, {}).get('weight', 0)
        weight_h = h.get_edge_data(*edge, {}).get('weight', 0)
        mins += min(weight_g, weight_h)
        maxs += max(weight_g, weight_h)

    return round((mins / maxs), 3)

def graph_metrics(G):
    
    edge_weights = nx.get_edge_attributes(G,'weight')
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    possible_edges  = (num_nodes*(num_nodes-1))/2
    
    components = nx.connected_components(G)
    largest_cc = max(components, key=len)
    subgraph   = G.subgraph(largest_cc)
    diameter   = nx.diameter(subgraph)
    
    density         = G.size(weight='weight')/possible_edges
    avg_degree      = sum(dict(G.degree(weight='weight')).values())/num_nodes
    transitivity    = nx.transitivity(G)
    avg_cc          = nx.average_clustering(G, weight='weight')
    
    print('Nodes:', num_nodes, 'Edges:', num_edges)
    print('Is connected?', nx.is_connected(G))
    print("Diameter LCC:", diameter)
    print('Density:', round(density, 4))
    print('Avg. degree:', round(avg_degree, 4))
    print('Transitivity:', round(transitivity, 4))
    print('Avg. CC:', round(avg_cc, 4))

def get_significant_edges(scores, metadata):

    x = scores

    stats_cols = ['edge', 'F', 'pvalue', 'significant', 'tukey',
                  'c0_mean', 'c1_mean', 'c2_mean']
    edges_stats = pd.DataFrame(columns=stats_cols)
    edges_stats = edges_stats.set_index('edge')

    for edge in x:

        data_c0 = x.loc[metadata['cluster'] == 0][edge].dropna().values
        data_c1 = x.loc[metadata['cluster'] == 1][edge].dropna().values
        data_c2 = x.loc[metadata['cluster'] == 2][edge].dropna().values

        fvalue, pvalue = stats.f_oneway(data_c0, data_c1, data_c2)

        if pvalue < 0.001:
            s = '***'
        elif pvalue < 0.01:
            s = '**'
        elif pvalue < 0.05:
            s = '*'
        else:
            s = 'ns'

        if s != 'ns':

            # perform multiple pairwise comparison (Tukey HSD)
            m_comp = pairwise_tukeyhsd(endog=x[edge],
                                       groups=metadata['cluster'],
                                       alpha=0.05)
            
            reject = m_comp.reject
            codes  = ['a', 'b', 'c']
            tukey_HSD = [j for i, j in zip(reject, codes) if i]

            c0_mean = round(scores.loc[metadata['cluster'] == 0][edge].dropna().values.mean(), 4)
            c1_mean = round(scores.loc[metadata['cluster'] == 1][edge].dropna().values.mean(), 4)
            c2_mean = round(scores.loc[metadata['cluster'] == 2][edge].dropna().values.mean(), 4)
            
            pvalue = '{:.2E}'.format(pvalue)

            edges_stats.loc[edge] = [round(fvalue, 2), pvalue, s, tukey_HSD, c0_mean, c1_mean, c2_mean]
    
    edges_stats = edges_stats.loc[edges_stats['tukey'].astype('str') != '[]']
    edges_stats.to_csv(f'results/significant_edges.csv')
            
    return edges_stats
    
def make_gene_list(edges_stats, net):
    
    c0_nodes = []; c1_nodes = []; c2_nodes = []

    for index, row in edges_stats.iterrows():

        if 'a' in row['tukey']:

            if row['c0_mean'] < row['c1_mean']:
                c0_nodes.append(index.split('-')[0])
                c0_nodes.append(index.split('-')[1])

            elif row['c1_mean'] < row['c0_mean']:
                c1_nodes.append(index.split('-')[0])
                c1_nodes.append(index.split('-')[1])

        if 'b' in row['tukey']:

            if row['c0_mean'] < row['c2_mean']:
                c0_nodes.append(index.split('-')[0])
                c0_nodes.append(index.split('-')[1])

            elif row['c2_mean'] < row['c0_mean']:
                c2_nodes.append(index.split('-')[0])
                c2_nodes.append(index.split('-')[1])

        if 'c' in row['tukey']:

            if row['c1_mean'] < row['c2_mean']:
                c1_nodes.append(index.split('-')[0])
                c1_nodes.append(index.split('-')[1])

            elif row['c2_mean'] < row['c1_mean']:
                c2_nodes.append(index.split('-')[0])
                c2_nodes.append(index.split('-')[1])
    
    # Save significant nodes/genes
    with open('results/cluster0_nodes_significant.txt','w') as f:
          for line in list(set(c0_nodes)):
            f.write(f"{line}\n")

    with open('results/cluster1_nodes_significant.txt','w') as f:
          for line in list(set(c1_nodes)):
            f.write(f"{line}\n")

    with open('results/cluster2_nodes_significant.txt','w') as f:
          for line in list(set(c2_nodes)):
            f.write(f"{line}\n")

    signf_nodes = set(list(c0_nodes) + list(c1_nodes) + list(c2_nodes))
    with open('results/nodes_significant.txt','w') as f:
          for line in signf_nodes:
            f.write(f"{line}\n")

    # Save original nodes to compare
    original_nodes = list(net.nodes())
    with open('results/original_nodes.txt','w') as f:
          for line in original_nodes:
            f.write(f"{line}\n")
            
            
    return c0_nodes, c1_nodes, c2_nodes, original_nodes

def enrichment_analysis(databases, c0_nodes, c1_nodes, c2_nodes, original_nodes, edges_significant):
    
    '''
    databases: list of GSEAPy available gene sets
    '''
    
    for gene_sets in databases:
        
        dfs = []
        i = 0
        for nodes in [c0_nodes, c1_nodes, c2_nodes]:
        
            if len(nodes) != 0:

                enr = gp.enrichr(gene_list=nodes, gene_sets=[gene_sets],
                                 background=original_nodes,
                                 organism='human', outdir=None)

                gsea_results = enr.results
                gsea_results['cluster'] = f'Cluster {i}'
                signf = gsea_results.loc[gsea_results['Adjusted P-value'] < 0.05]
                dfs.append(signf)

            i += 1
        
    results = pd.concat(dfs)
    results.to_csv(f'results/{gene_sets}_not_filtered.csv', index=None)
    results.reset_index(inplace=True)

    indexes = []
    edges_found = []
    for index, row in results.iterrows():
        genes_row = set(row['Genes'].split(';'))
        for edge in edges_significant:
            genes_edge = set(edge.split('-'))
            intersection = genes_row.intersection(genes_edge)
            if len(intersection) == 2:
                indexes.append(index)
                edges_found.append(edge)

    results_edges = results.loc[indexes]
    results_edges['Edges'] = edges_found

    columns = list(results_edges.columns)
    columns.remove('Edges')

    results_edges = results_edges.groupby(columns)['Edges'].apply(lambda x: ';'.join(x.astype(str))).reset_index()
    results_edges.drop(columns=['index'], inplace=True)
    results_edges.to_csv(f'results/{gene_sets}.csv', index=None)

    return results_edges

def stats_numerical(data, feat_cluster, round_mean, round_std):
    
    # Prepare output dataframe
    statistics_features = pd.DataFrame(columns=['feature', 'F', 'pvalue', 'significant',
        'tukey_HSD', 'cluster1', 'cluster2', 'cluster3'])
    statistics_features = statistics_features.set_index('feature')
    
    for c in data.columns.drop(feat_cluster):
    
        data = data.astype('float64')
        
        # Select data from each cluster
        data_c0 = data.loc[data[feat_cluster] == 0][c].dropna().values
        data_c1 = data.loc[data[feat_cluster] == 1][c].dropna().values
        data_c2 = data.loc[data[feat_cluster] == 2][c].dropna().values

        mean_c0 = round(data_c0.mean(), round_mean)
        mean_c1 = round(data_c1.mean(), round_mean)
        mean_c2 = round(data_c2.mean(), round_mean)

        std_c0 = round(data_c0.std(), round_std)
        std_c1 = round(data_c1.std(), round_std)
        std_c2 = round(data_c2.std(), round_std)

        point_c0 = f'{mean_c0} ± {std_c0}'
        point_c2 = f'{mean_c2} ± {std_c2}'
        point_c1 = f'{mean_c1} ± {std_c1}'
        
        # Calculate ANOVA statistics
        fvalue, pvalue = stats.f_oneway(data_c0, data_c1, data_c2)
        
        # p-value codes
        if pvalue < 0.001:
            s = '***'
        elif pvalue < 0.01:
            s = '**'
        elif pvalue < 0.05:
            s = '*'
        else:
            s = 'ns'
            
            
        # perform multiple pairwise comparison (Tukey HSD)       
        m_comp = pairwise_tukeyhsd(endog=data[c], groups=data[feat_cluster], alpha=0.01)
        tukey_HSD = [j for i, j in zip(m_comp.reject, ['a', 'b', 'c']) if i]
        
        # Save results
        statistics_features.loc[c] = [round(fvalue, 2), round(pvalue, 4), s, tukey_HSD,
                                        point_c0, point_c1, point_c2]
    
    return statistics_features

def load_adnimerge_data(data, feat_cluster):

    adnimerge = pd.read_csv('data/ADNI/ADNIMERGE.csv', index_col='PTID', low_memory=False)
    adnimerge['DX_bl'].replace({'AD':'Dementia', 'EMCI':'MCI', 'LMCI':'MCI'}, inplace=True)
    
    soc = adnimerge[['PTGENDER', 'PTEDUCAT', 'AGE', 'PTETHCAT', 'APOE4', 'DX_bl', 'DX']]
    cog = adnimerge[['MMSE_bl', 'CDRSB_bl']]
    bio = adnimerge[['ABETA_bl', 'TAU_bl', 'PTAU_bl', 'AV45_bl', 'FDG_bl',
                           'WholeBrain_bl', 'Ventricles_bl', 'MidTemp_bl', 'Hippocampus_bl',
                           'Fusiform_bl', 'Entorhinal_bl', 'ICV_bl']]

    # Select clusters samples
    soc = soc[~soc.index.duplicated(keep='first')].loc[data.index]
    bio = bio[~bio.index.duplicated(keep='first')].loc[data.index]
    cog = cog[~cog.index.duplicated(keep='first')].loc[data.index]

    # Normalize MRI features
    icv_mean = bio['ICV_bl'].mean()
    
    bio['WholeBrain_bl']  = bio['WholeBrain_bl']/icv_mean.astype(float)
    bio['Ventricles_bl']  = bio['Ventricles_bl']/icv_mean.astype(float)
    bio['MidTemp_bl']     = bio['MidTemp_bl']/icv_mean.astype(float)
    bio['Hippocampus_bl'] = bio['Hippocampus_bl']/icv_mean.astype(float)
    bio['Fusiform_bl']    = bio['Fusiform_bl']/icv_mean.astype(float)
    bio['Entorhinal_bl']  = bio['Entorhinal_bl']/icv_mean.astype(float)

    # Replace non-numerical values 
    bio['ABETA_bl'] = bio['ABETA_bl'].replace({'<200': 200, '>1700':1700}).astype(float)
    bio['TAU_bl']   = bio['TAU_bl'].replace({'<8': 8}).astype(float)
    bio['PTAU_bl']  = bio['PTAU_bl'].replace({'<8': 8}).astype(float)

    bio = pd.concat([bio, data[feat_cluster]], axis=1, join='inner')
    soc = pd.concat([soc, data[feat_cluster]], axis=1, join='inner')
    cog = pd.concat([cog, data[feat_cluster]], axis=1, join='inner')

    print(cog.shape[0], soc.shape[0], bio.shape[0])

    return soc, cog, bio

def stats_categorical(data, feat_cluster):

    # Prepare output dataframe
    statistics_features = pd.DataFrame(columns=['feature', 'chstat', 'pvalue', 'dof',
                                                'significant', 'cluster1', 'cluster2', 'cluster3'])

    statistics_features = statistics_features.set_index('feature')

    for c in data.columns.drop(feat_cluster):

        if c == 'DX':
            data[c] = data[c].replace({'MCI':'not CN', 'Dementia':'not CN'})
        elif c == 'DX_bl':
            data[c] = data[c].replace({'EMCI':'not CN', 'LMCI':'not CN', 'AD':'not CN', 'Dementia':'not CN'})
        elif c == 'APOE4':
            data[c] = data[c].replace({0:'non carrier', 1:'carrier', 2:'carrier'})

        contigency_norm = pd.crosstab(data[c], data[feat_cluster], normalize='columns')
        contigency      = pd.crosstab(data[c], data[feat_cluster])

        vars = {'PTGENDER':'Female',
                'PTETHCAT':'Not Hisp/Latino',
                'APOE4':'non carrier',
                'DX_bl':'CN', 'DX':'CN',
                '_E2/E2': 1,
                '_E2/E3': 1,
                '_E2/E4': 1,
                '_E3/E4': 1,
                '_E4/E4': 1}

        per_c0 = round(contigency_norm.loc[vars[c]][0]*100, 2)
        per_c1 = round(contigency_norm.loc[vars[c]][1]*100, 2)
        per_c2 = round(contigency_norm.loc[vars[c]][2]*100, 2)

        n_c0 = contigency.loc[vars[c]][0]
        n_c1 = contigency.loc[vars[c]][1]
        n_c2 = contigency.loc[vars[c]][2]

        point_c0 = f'{n_c0} ({per_c0}%)'
        point_c1 = f'{n_c1} ({per_c1}%)'
        point_c2 = f'{n_c2} ({per_c2}%)'

        # Chi-squared stats
        chstat, pvalue, dof, expected = chi2_contingency(contigency)

        chstat = round(chstat, 2)
        pvalue = round(pvalue, 8)

        # p-value codes
        if pvalue < 0.001:
            s = '***'
        elif pvalue < 0.01:
            s = '**'
        elif pvalue < 0.05:
            s = '*'
        else:
            s = 'ns'

        c = f'{c} ({vars[c]})'

        # Save results
        statistics_features.loc[c] = [chstat, pvalue, dof, s, point_c0, point_c1, point_c2]

    return statistics_features

