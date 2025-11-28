import torch
import numpy as np
from Bio.PDB import PDBParser
import os


ATOM_MAP = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

class ProteinFeaturizer:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def get_atom_features(self, element):
    
        idx = ATOM_MAP.get(element, 4) 
        one_hot = np.zeros(5)
        one_hot[idx] = 1.0
        return one_hot

    def process_pdb(self, pdb_path):
        structure = self.parser.get_structure('protein', pdb_path)
        
        node_features = []  
        coords = []        

        for atom in structure.get_atoms():
            if atom.get_parent().get_resname() == 'HOH':
                continue
                
            coords.append(atom.get_coord())
            
            element = atom.element
            if not element: 
                element = atom.get_name()[0]
            
            node_features.append(self.get_atom_features(element))

        X = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        Pos = torch.tensor(np.array(coords), dtype=torch.float32)
        
        return X, Pos

    def build_adjacency(self, coords, threshold=4.5):
        dist_matrix = torch.cdist(coords, coords)
        
        adj_matrix = (dist_matrix < threshold) & (dist_matrix > 0)
        
        return adj_matrix.float()

if __name__ == "__main__":
    pdb_file = "data/pdb/pdb1crn.ent" 
    
    if os.path.exists(pdb_file):
        featurizer = ProteinFeaturizer()
        
        print("Processing Protein Graph...")
        
        node_feats, coordinates = featurizer.process_pdb(pdb_file)
        
        adjacency = featurizer.build_adjacency(coordinates)
        
        print("\n" + "="*30)
        print("SYNAPSE TENSORS READY")
        print("="*30)
        print(f"Node Features (X): {node_feats.shape}")
        print(f"   -> Example (First Atom): {node_feats[0]}") 
        print(f"Coordinates (Pos): {coordinates.shape}")
        print(f"Adjacency Matrix: {adjacency.shape}")
        print(f"Total Connections (Edges): {adjacency.sum().item()}")
        print("="*30)
        
        if torch.cuda.is_available():
            print("\nCUDA Check: Moving tensors to GPU...")
            node_feats = node_feats.to('cuda')
            print("Success! Tensors are now on NVIDIA GPU.")
        else:
            print("\nWarning: CUDA not available. Running on CPU.")
            
    else:
        print("Error: PDB file not found. Run 01_data_loader.py first.")
