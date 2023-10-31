import requests
import json
import networkx as nx
import mygene
import os, io
import numpy as np
import pandas as pd

def get_snap(genes, remove_components):

    print(len(genes))

    G = nx.read_edgelist('data/networks/PPT-Ohmnet_tissues-combined.edgelist', nodetype=int, data=(('tissue', str),))

    tissues_edgelist = pd.read_csv('data/networks/PPT-Ohmnet_tissues-combined.edgelist', sep='\t')
    brain_specific = tissues_edgelist[tissues_edgelist['tissue'] == 'brain']
    brain_specific.to_csv('data/networks/PPT-Ohmnet_tissues-brain.edgelist', sep='\t', index=False)
    G_brain = nx.read_edgelist('data/networks/PPT-Ohmnet_tissues-brain.edgelist', nodetype=int, data=(('tissue', str),))

    # Genes in PPT-Ohmnet are Entrez IDs, it is necessary to convert them to gene Symbols.
    mg = mygene.MyGeneInfo()
    out = mg.querymany(genes, scopes='symbol', fields='entrezgene', species='human', verbose=False)

    entrezgenes = []
    mapping = {}
    for o in out:
        if 'entrezgene' in o:
            entrezgenes.append(int(o['entrezgene']))
            mapping[int(o['entrezgene'])] = o['query']

    A_brain_frozen = G_brain.subgraph(entrezgenes)
    A_brain = nx.Graph(A_brain_frozen)
    original = A_brain.number_of_nodes()

    if remove_components == True:
        # Delete nodes from components with less than 5 nodes
        nodes_to_remove = []
        for component in list(nx.connected_components(A_brain)):
            if len(component)<5:
                for node in component:
                    A_brain.remove_node(node)

    # Remove self-loops
    A_brain.remove_edges_from(list(nx.selfloop_edges(A_brain)))

    largest = A_brain.number_of_nodes()
    lost = original - largest
    lost_percent = round((lost/original), 4)

    print()
    print('SNAP')
    print('Whole network:', original, 'nodes')
    print('Biggest connected component:', largest, 'nodes')
    print('Percentage of lost genes/nodes:', lost, f'({lost_percent*100}%)')

    A_brain_relabeled = nx.relabel_nodes(A_brain, mapping)
    nx.write_edgelist(A_brain_relabeled, f'data/networks/PPI_SNAP_brain_{remove_components}.edgelist')

    return A_brain_relabeled


