import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import networkx as nx

class UKBiobankDataset(Dataset):
    '''
    eid_list is the list of sample IDs;
    clinical_data is the processed dataframe of clinical predictors;
    genomics_data is the processed dataframe of genomics predictors;
    metabolomics_data is the processed dataframe of metabolomics predictors;
    proteomics_data is the processed dataframe of proteomics predictors;
    y is the dataframe of survival times;
    e is the dataframe of event indicators
    '''
    def __init__(self, eid_list, clinical_data, genomics_data, metabolomics_data, proteomics_data, y, e, ppi=None):
        super(UKBiobankDataset, self).__init__()
        self.eid_list = eid_list
        self.clinical_array = clinical_data.values
        self.genomics_array = genomics_data.values
        self.metabolomics_array = metabolomics_data.values
        self.proteomics_array = proteomics_data.values
        self.y_array = y.values
        self.e_array = e.values
    
        # Protein-protein interaction network
        self.ppi = ppi
        if self.ppi is not None:
            self.ppi = self.ppi.rename(columns={'protein1_name': 'from', 'protein2_name': 'target'})
            _, self.node_list_ppi, self.node_to_idx_ppi, self.edge_index_ppi = self._build_network(node_df=proteomics_data, edge_df=self.ppi)
            
    ''' Build biological interaction networks '''
    def _build_network(self, node_df, edge_df):
        # Create undirected graph
        G = nx.Graph()
        
        # Add nodes
        node_list = node_df.columns.tolist()
        G.add_nodes_from(node_list)
        
        # Add edges
        for _, row in edge_df.iterrows():
            G.add_edge(row['from'], row['target'])

        # Create node index mapping
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        
        # Create edge index tensor
        edges = [(node_to_idx[edge[0]], node_to_idx[edge[1]]) for edge in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return G, node_list, node_to_idx, edge_index
    
    
    def __getitem__(self, idx):
        eid_tensor = torch.tensor(self.eid_list[idx], dtype=torch.int).unsqueeze(0) # [batch_size, 1]
        
        clinical_tensor = torch.tensor(self.clinical_array[idx], dtype=torch.float)
        genomics_tensor = torch.tensor(self.genomics_array[idx], dtype=torch.float)
        metabolomics_tensor = torch.tensor(self.metabolomics_array[idx], dtype=torch.float)
        proteomics_tensor = torch.tensor(self.proteomics_array[idx], dtype=torch.float)
        
        if self.ppi is not None:
            node_features = self.proteomics_array[idx, :]
            node_features = torch.tensor(node_features, dtype=torch.float).view(-1, 1)
            proteomics_graph_tensor = Data(x=node_features, edge_index=self.edge_index_ppi)
        
        y_tensor = torch.tensor(self.y_array[idx], dtype=torch.float)
        e_tensor = torch.tensor(self.e_array[idx], dtype=torch.float)

        if self.ppi is not None:
            return eid_tensor, clinical_tensor, genomics_tensor, metabolomics_tensor, proteomics_tensor, proteomics_graph_tensor, y_tensor, e_tensor
        else:
            return eid_tensor, clinical_tensor, genomics_tensor, metabolomics_tensor, proteomics_tensor, y_tensor, e_tensor

    def __len__(self):
        return len(self.y_array)
