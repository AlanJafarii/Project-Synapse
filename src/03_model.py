import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser 

from importlib import import_module
featurizer_module = import_module("02_featurizer") 

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        
        self.W1 = nn.Linear(input_dim, hidden_dim)
        
        self.W2 = nn.Linear(hidden_dim, output_dim)
        
        print(f"Model Initialized: Input={input_dim} -> Hidden={hidden_dim} -> Output={output_dim}")

    def forward(self, x, adj):
        """
        x: ویژگی‌های اتم‌ها (N, 5)
        adj: ماتریس همسایگی (N, N)
        """
    
        h = self.W1(x) 
        
      
        h = torch.mm(adj, h)
        
        h = F.relu(h)
        
        out = self.W2(h)
        
        return out

if __name__ == "__main__":
    pdb_file = "data/pdb/pdb1crn.ent"
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    adj = feat.build_adjacency(pos)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    adj = adj.to(device)
    
    print("\n" + "="*30)
    print("INITIALIZING SYNAPSE AI ENGINE")
    print("="*30)
    
  
    model = SimpleGNN(input_dim=5, hidden_dim=16, output_dim=1).to(device)
    
    with torch.no_grad(): 
        prediction = model(x, adj)
    
    print("\n--- Model Output Analysis ---")
    print(f"Input Shape: {x.shape} (Atoms, Features)")
    print(f"Output Shape: {prediction.shape} (Atoms, Predicted_Value)")
    
    print("\nFirst 5 Atoms Predictions (Random Weights):")
    print(prediction[:5].cpu().numpy())
    
    print("\nStatus: The Brain is ALIVE and processing geometry!")
