import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pingouin import partial_corr
import networkx as nx
import networkx.algorithms.community as nx_comm
from itertools import permutations

def group_by_test(metadata, df):
   """Function to group the scores measuring the same functions"""

   test_ls = metadata['Label'].unique()
   
   #Iterate each test
   for test in test_ls:
      variable_ls = metadata[metadata['Label']==test]['ADNI column'].tolist()
      df[test] = df[variable_ls].sum(axis=1)
      
   return df


def zscores_means(X, dx, metadata_path):
   """Function to generate a table with the means by cognitive domain and diagnostic group"""
   #Compute the test means
   means_df = X.mean(axis=0).to_frame()
   means_df.columns = ["Mean"]
   means_df['Label'] = means_df.index

   #Import the test-cognitive domain relations    
   metadata = pd.read_csv(metadata_path, sep=";", 
                           usecols = ['Label', 'Cognitive Domain'])

   means_df = means_df.join(metadata.set_index('Label'))[['Cognitive Domain', 'Mean']]

   #Compute NC domain means
   means_df = means_df.groupby(['Cognitive Domain'])['Mean'].mean().to_frame()
   means_df.reset_index(inplace=True)

   #Add diagnostic group
   means_df["Diagnostic"] = dx
   
   return(means_df)

def par_corr(data_df):
   """
   Compute partial pairwise correlation of columns. 
   When a pair of columns are picked, then all other columns are treated as control variables. 
   
   @param data_df DataFrame
   @return DataFrame, whose data is a symmetric matrix
   """ 
   
   n = data_df.shape[1] #total number of tests
   mat = np.empty((n, n)) #empty matrix to store results 
   np.fill_diagonal(mat, 1) #diagonal elements have correlation equal to 1.0
   count_neg = 0 #count negative correlations
   
   for i in range(n):
      for j in range(i + 1, n):
         #get columns names
         x = data_df.columns[i]
         y = data_df.columns[j]
         xy_colnames = [data_df.columns[index] for index in [i,j]]
         covar = [ var for var in data_df.columns if var not in xy_colnames]
         
         #partial correlation
         corr_df = partial_corr(data=data_df, x=x, y=y, covar=covar, method='spearman') #partial correlation stats
         p_value = corr_df.iloc[0]['p-val']
         corr = corr_df.iloc[0]['r'] #get partial correlation value
         #if p_value < 0.05:
         #    corr = corr_df.iloc[0]['r'] #get partial correlation value
         #else:
         #    corr = 0

         #store results
         mat[i, j] = corr
         mat[j, i] = corr
         
   return pd.DataFrame(mat, index=data_df.columns, columns=data_df.columns)

def plot_adjacency_mx(mx, variable, cluster, battery_name):
   fig = plt.figure(figsize=(10,8))
   
   sns.heatmap(data=mx, annot=False, cmap="Spectral_r", vmin=-1, vmax=1)

   #add title
   plt.title('Adjacency matrixes ' + variable + '-cluster ' + str(cluster) + ' (' + battery_name + ')')
   
   filename = "./Genetic-Clusters/Results_weights/" + battery_name + "/" + variable + "/Figures/matrix_cluster" + str(cluster) + ".svg"
   
   plt.savefig(filename, format="svg")
   plt.close(fig) #do not show figure 
   #plt.show()

def bootstrap(X, n_size, n_rep):
   """Function to generate an array of random adjacency matrices obtained trough randomly selecting patients from 
   each cluster (resampling with repositioning)
   Parameters:
   - X: df with the normalizad results for each test and patient
   - n_size: size of the new samples
   - n_rep: number of samples
   Output:
   - Array with all the adjacencies matrices obtained from the new samples"""
   #Parameter of the adjacency matrix
   n_rows = X.shape[1]
   n_cols = X.shape[1]
   labels = X.columns 
   
   #Create array to store results
   adj_matrices = np.zeros((n_rep, n_rows, n_cols))
   
   #Convert dataframe into numpy array
   #to gain computational efficiency
   #arr = X.to_numpy()
   
   print("Running bootstrap...")
   for i in range(n_rep):
      sample = X.sample(n = n_size, replace = True)
      #sample = pd.Dataframe(sample_arr, columns=labels, index=labels)
      #generate adjacency matrix
      mx = par_corr(sample)
      adj_matrices[i] = mx.to_numpy()
   
   return adj_matrices

def adj_matrix(matrices,labels):
   """Generate a matrix with the mean values for each pair"""
   
   #Create an array to store the element-wise sum of the matrices
   sum_mx = np.zeros(matrices[0].shape)
   
   # Loop over each matrix and add it to the sum matrix
   for mx in matrices:
      sum_mx += mx
   
   # Divide the sum matrix by the number of matrices to get the mean
   mean_mx = sum_mx / len(matrices)
   
   adj_mx = pd.DataFrame(mean_mx, columns=labels, index=labels)
   
   return adj_mx

def adj_matrix(matrices,labels):
   """Generate a matrix with the mean values for each pair"""
   
   #Create an array to store the element-wise sum of the matrices
   sum_mx = np.zeros(matrices[0].shape)
   
   #Initializate auxiliary variables
   sum_values = np.zeros(matrices[0].shape)
   n_values = np.zeros(matrices[0].shape)
   
   # Loop over each matrix and add it to the sum matrix
   for mx in matrices:
      # Iterate the matrix
      for i in range(mx.shape[0]):
         for j in range(mx.shape[1]):
               # if the value is not null, add it:
               if not np.isnan(mx[i,j]):
                  sum_values[i,j] += mx[i,j]
                  n_values[i,j] += 1

   
   # Get the mean
   mean_mx = np.divide(sum_values, n_values)
   
   adj_mx = pd.DataFrame(mean_mx, columns=labels, index=labels)
   
   return adj_mx

def cognitive_network(mx):
   """Function to remove diagonal elements and convert scores into absolute values. It returns a networkx graph"""

   for i in range(mx.shape[1]): #iterate matrix elements
      colname = mx.columns[i]
      #Remove diagonal elements
      mx[colname] =np.where((mx[colname]==1.0) | (mx[colname].isnull()),0, mx[colname]) 
      #Convert negative correlations in positive ones
      mx[colname] = np.where((mx[colname]<0),-1*mx[colname], mx[colname]) 
   
   #Create graph from adjacency matrix
   g = nx.from_numpy_array(mx.to_numpy())
   
   return g

def node_attributes(metadata_path, graph, X): 
   """Function to add attributes to the nodes of the graph"""
   
   attribute_ls = ['Node', 'Label', 'Test', 'Cognitive Domain']
   
   #Import node metadata
   metadata_df = pd.read_csv(metadata_path, sep=";", 
                              usecols = attribute_ls)
   metadata_df = metadata_df[attribute_ls].drop_duplicates()
   
   #Add attributes
   for attribute in attribute_ls:
      nx.set_node_attributes(graph, dict(zip(metadata_df.Node, metadata_df[attribute])), name=attribute)
      
   #add mean z-score to each node 
   means = X.mean(axis=0).tolist()
   nx.set_node_attributes(graph, dict(zip(metadata_df.Node, means)), name='Zscore mean')

def centrality(cm_df,graph,cluster, metric, index):
    
   """Function to compute a table with the different centrality measures for each node."""
   
   #compute edges distances based on weights
   g_distance_dict = {(e1, e2): 1/weight for e1, e2, weight in graph.edges(data='weight')}
   nx.set_edge_attributes(graph, g_distance_dict, 'distance')
   
   if metric == 'DC': 
      cm = [graph.degree(n, weight='weight') for n in graph.nodes()] #degree centrality
      cm.insert(0, 'DC')
      cm.insert(0, cluster)
   
   elif metric == 'BC':
      cm = list(nx.betweenness_centrality(graph, weight='distance').values()) #betweenness centrality
      cm.insert(0, 'BC')
      cm.insert(0, cluster)

   cm_df.loc[index] = cm
      
   return cm_df

def global_efficiency_weighted(G):
   """Function to compute global efficiency in a weighted graph"""
   n = len(G)
   denom = n * (n - 1)
   if denom != 0:
      shortest_paths = dict(nx.all_pairs_dijkstra(G, weight = 'distance'))
      g_eff = sum(1./shortest_paths[u][0][v] if shortest_paths[u][0][v] !=0 else 0 for u, v in permutations(G, 2)) / denom
   else:
      g_eff = 0
   return g_eff


def global_metrics(gm_df, graph, cluster, index):
    
   """Function to compute a table with some global metrics of the graphs. 
   It returns a pandas DataFrame object."""
   
   #GLOBAL METRICS
   number_nodes = graph.number_of_nodes() #number of nodes
   number_edges = graph.number_of_edges() #number of edges
   shortest = nx.shortest_path_length(graph, weight="weight") #matrix of shortest paths

   if nx.is_connected(graph):
      ecc = nx.eccentricity(graph, sp=dict(shortest))
      diameter = nx.diameter(graph) #diameter of graph
   else:
      diameter = float("nan")
      
   possible_edges = (number_nodes*(number_nodes-1))/2
   density = graph.size(weight='weight')/ possible_edges
   avg_degree = sum(dict(graph.degree(weight='weight')).values())/graph.number_of_nodes() #average degree
   transitivity = nx.transitivity(graph) #transitivity of graph
   avg_cc = nx.average_clustering(graph,weight='weight') #average clustering coefficient
   
   gm_df.loc[index] = [cluster, number_nodes, number_edges,
                       diameter, density, avg_degree, transitivity, avg_cc]
   
   return gm_df

def bootstrap_metrics(node_ls, matrices, cluster, gm_df, dc_df, bc_df):
   n_rep = len(matrices)
   index = 0 + n_rep*cluster #row index
   labels = node_ls
   
   print("Computing metrics for cluster ", cluster)
   for i in range(n_rep):
      if i%5 == 0: 
         print("Running repetition ", i)
      sample = matrices[i]
      mx = pd.DataFrame(sample, columns=labels, index=labels)
      
      #generate graph
      graph = cognitive_network(mx)
      #compute global metrics
      gm_df = global_metrics(gm_df, graph, cluster, index)
      #Compute local metrics
      dc_df = centrality(dc_df,graph,cluster, 'DC', index)
      bc_df = centrality(bc_df,graph,cluster, 'BC', index)
      
      index+=1
   
   return gm_df, dc_df, bc_df


def community_detection(graph, algorithm):
      
   if algorithm == "Louvain": 
      partition = nx_comm.louvain_communities(graph, weight='weight',seed=0)

   elif algorithm == "Greedy":
      partition = nx_comm.greedy_modularity_communities(graph, weight='weight')

   elif algorithm == "Bisection":
      partition = nx_comm.kernighan_lin_bisection(graph, weight='weight', seed=0)

   elif algorithm == "Label Propagation":
      partition = list(nx_comm.asyn_lpa_communities(graph, weight='weight', seed=0))

   else:
      print("This algorithm is not implemented. Please, try again.")
   
   return partition

def color_communities(graph, partition):
    
   colors = {} #create empty node dictionary
   
   #Convert set of partitions into node dictionary
   for i in range(len(partition)): #iterate for each group
      for node in partition[i]:
         colors[node] = i
         
   #Sort dictionary by keys
   node_list = list(colors.keys())
   node_list.sort()
   sorted_colors = {i: colors[i] for i in node_list}
         
   return list(sorted_colors.values())

def draw_graph_communities(graph, partition, test_labels, pos, battery_name, variable, cluster, algorithm):

   #Plot
   fig = plt.figure(figsize=(10,8))

   #node colors
   colors = color_communities(graph, partition)

   #get edges weights
   weights = list(nx.get_edge_attributes(graph,'weight').values())

   if test_labels is not None: 
      nx.draw(G=graph, pos=pos, labels=test_labels, with_labels=True, node_color=colors, cmap=plt.cm.Spectral,
            edge_color=weights, edge_cmap=plt.cm.Greys, width=[ x*10 for x in weights])

   else:
      nx.draw(G=graph, pos=pos, with_labels=True, node_color=colors, cmap=plt.cm.Spectral,
            edge_color=weights, edge_cmap=plt.cm.Greys, width=[ x*10 for x in weights])
   

   #add title 
   plt.title('Cognitive network ' + algorithm + "algorithm, " + variable + '-cluster ' + str(cluster) + ' (' + battery_name + ')')
   
   #save figure
   figure_path = "./Genetic-Clusters/Results/" + battery_name + "/" + variable +"/Figures/graph_" + algorithm + "_cluster-" + str(cluster) +".svg"


   plt.savefig(figure_path, format="svg")
   plt.close(fig)
   #plt.show()

def metrics(graph):
   """Function to create a dictionary with all the metrics computed for a community"""
   
   metrics_dict = {}
   
   #GLOBAL METRICS
   #Compute the number of nodes 
   metrics_dict['NNodes'] = graph.number_of_nodes()
   #Compute the number of edges 
   metrics_dict['NEdges'] = graph.number_of_edges()
   #Compute the diameter of the graph
   shortest = nx.shortest_path_length(graph, weight="weight")
   ecc = nx.eccentricity(graph, sp=dict(shortest))
   metrics_dict['Diameter'] = nx.diameter(graph, e=ecc)
   #Compute the density of the graph
   metrics_dict['Density'] = nx.density(graph)
   #Compute the average degree of the network  
   metrics_dict['AvDegree'] = sum(dict(graph.degree(weight='weight')).values())/graph.number_of_nodes()
   #Compute the transitivity of the graph
   metrics_dict['Transitivity'] = nx.transitivity(graph)
   #Compute the average clustering coefficient
   metrics_dict['AvCC'] = nx.average_clustering(graph,weight='weight')
   #Compute the average global efficiency (shortest path)
   metrics_dict['AvGE'] = nx.global_efficiency(graph)
   
   #TESTS BELONGING TO THE COMMUNITY
   metrics_dict['Tests'] = list(dict(graph.nodes(data="Label")).values())
   
   return metrics_dict

def community_metrics(graph, partition):
   
   """Function to create a dataframe with all the metrics computed for each of the communities"""
   
   domains_list = list(dict(graph.nodes(data="Cognitive Domain")).values())
   domains = [*set(domains_list)] #unique list of domains
   
   df = pd.DataFrame(columns = ['Index','NNodes', 'NEdges', 'Diameter', 'Density', 'AvDegree', 
                              'Transitivity','AvCC', 'AvGE', 'Tests'] + domains) #empty dataframe
   
   for i in range(len(partition)): #iterate for each community
      #create subgraph for this community
      subgraph = graph.subgraph(partition[i]) 
      #compute metrics for the subgraph 
      metrics_dict = metrics(subgraph)
      
      #community index
      metrics_dict['Index'] = i
      
      #representation of each neurocognitive domain
      domains_list_community = list(dict(subgraph.nodes(data="Cognitive Domain")).values())
      
      for domain in domains: 
         #domain_count_total = domains_list.count(domain)
         domain_count_community = domains_list_community.count(domain)
         domain_count_total = len(domains_list_community)
         metrics_dict[domain] = domain_count_community/domain_count_total #percentage of representation 
      
      #introduce metrics in new row
      df = df.append(metrics_dict, ignore_index=True)
      
   return df

def domains_rep(graph, partition, battery_name, algorithm, variable, cluster):

   domains_ls = np.unique(list(nx.get_node_attributes(graph, 'Cognitive Domain').values())).tolist()
   columns = ['Index'] + domains_ls

   domains = community_metrics(graph, partition) #get data
   nodomains = domains.loc[:, ~domains.columns.isin(columns)]
   path = "./Genetic-Clusters/Results/" + battery_name + "/" + variable + "/" + algorithm 
   if not os.path.exists(path):
      os.makedirs(path)
   filepath= path + "/metrics_cluster-" + str(cluster) + ".csv" 
   nodomains.to_csv(filepath, index=False)

   #Reshape data with melt() function
   domains = domains[columns]
   domains_rs = domains.melt(id_vars=['Index'], var_name="Domain", value_name='Percentage')

   #send data to build stacked barplot in R
   filepath = path  + "/domains_cluster-" + str(cluster) + ".csv"  
   domains_rs.to_csv(filepath, index=False)


def plot_cognitive_network(G, pos, title):

   # nx.write_gexf(G, f"results/cognitive_profiles_{title}.gexf")

   nodelist = G.nodes()

   # print(G.nodes(data=True))

   # Node labels, sizes and colors 
   mapping = {'Attention':"#9999FF", 'Executive':"#CC99FF",
              'Language':"#F8766D", 'Memory':"#00BFC4",
              'Orientation':"#66CC99", 'Visuospatial':"#FFCC99"}
   
   node_labels = nx.get_node_attributes(G, 'Label')
   node_colors = [mapping[G.nodes[n]['Cognitive Domain']] for n in nodelist]

   # Edge widths and colors
   widths = nx.get_edge_attributes(G, 'weight')
   edge_colors = [G[u][v]['weight'] for u, v in G.edges]
   edge_weights = list(widths.values())
   edge_weights = [e * 10 for e in edge_weights]

   plt.figure(figsize=(14, 14))

   nx.draw_networkx_nodes(G, pos,
                           nodelist=nodelist,
                           node_size=2000,
                           node_color=node_colors,)

   nx.draw_networkx_edges(G, pos,
                           edgelist=widths.keys(),
                           width=edge_weights,
                           edge_color=edge_colors,
                           edge_cmap=plt.cm.gray_r)

   nx.draw_networkx_labels(G, pos=pos,
                           labels=node_labels,
                           font_color='black',
                           font_size=30)


   plt.box(False)
   plt.title(title, fontsize=30)
   plt.tight_layout()
   plt.savefig(f'figures/{title}.png', dpi=300)
   # plt.show()