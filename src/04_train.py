import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Bio.PDB import PDBParser

from importlib import import_module
featurizer_module = import_module("02_featurizer")
model_module = import_module("03_model")

def get_real_bfactors(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    bfactors = []
    
    for atom in structure.get_atoms():
        if atom.get_parent().get_resname() == 'HOH':
            continue
        bfactors.append(atom.get_bfactor())
    
    b_np = np.array(bfactors)
    b_norm = (b_np - b_np.min()) / (b_np.max() - b_np.min())
    
    return torch.tensor(b_norm, dtype=torch.float32).unsqueeze(1) 

if __name__ == "__main__":
    pdb_file = "data/pdb/pdb1crn.ent"
    print(f"Loading data from {pdb_file}...")
    
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    adj = feat.build_adjacency(pos)
    
    y_true = get_real_bfactors(pdb_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    x = x.to(device)
    adj = adj.to(device)
    y_true = y_true.to(device)
    
    model = model_module.SimpleGNN(input_dim=5, hidden_dim=32, output_dim=1).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\n" + "="*30)
    print("STARTING TRAINING LOOP")
    print("="*30)
    
    epochs = 200 
    
    for epoch in range(epochs):
        model.train() 
        
        optimizer.zero_grad()
        
        y_pred = model(x, adj)
        
        loss = criterion(y_pred, y_true)
        
        loss.backward() 
        optimizer.step() 
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.6f}")

    print("="*30)
    print("TRAINING COMPLETED")
    
    model.eval()
    with torch.no_grad():
        final_pred = model(x, adj)
        
    print("\nSample Results (Normalized B-factors):")
    print(f"{'Atom Idx':<10} | {'Prediction':<10} | {'Actual (True)':<10}")
    print("-" * 35)
    indices = [0, 1, 2, 100, 101, 102] 
    for i in indices:
        print(f"{i:<10} | {final_pred[i].item():.4f}     | {y_true[i].item():.4f}")
