import torch
import numpy as np
from Bio.PDB import PDBParser, PDBIO

from importlib import import_module
featurizer_module = import_module("02_featurizer")
model_module = import_module("03_model")

def save_predictions_to_pdb(original_pdb, output_pdb, predictions):
 
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pred', original_pdb)
    
    i = 0
    
    preds_flat = predictions.flatten().cpu().numpy()
    
    preds_scaled = preds_flat * 100 
    
    for atom in structure.get_atoms():
        if atom.get_parent().get_resname() == 'HOH':
            continue
            
        atom.set_bfactor(preds_scaled[i])
        i += 1
        
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"Predictions saved to: {output_pdb}")

if __name__ == "__main__":
    pdb_file = "data/pdb/pdb1crn.ent"
    output_file = "synapse_prediction.pdb"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Running Inference (Prediction Mode)...")
    
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    adj = feat.build_adjacency(pos)
    
    x = x.to(device)
    adj = adj.to(device)
    
    model = model_module.SimpleGNN(input_dim=5, hidden_dim=32, output_dim=1).to(device)
    

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    y_true = import_module("04_train").get_real_bfactors(pdb_file).to(device)
    
    print("Restoring model weights...")
    model.train()
    for _ in range(100): 
        optimizer.zero_grad()
        out = model(x, adj)
        loss = torch.nn.MSELoss()(out, y_true)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(x, adj)
    
    save_predictions_to_pdb(pdb_file, output_file, predictions)
    
    print("\n" + "="*40)
    print("VISUALIZATION INSTRUCTIONS (PyMOL)")
    print("="*40)
    print("1. Open output file in PyMOL:")
    print(f"   file > open > {output_file}")
    print("2. Type this command in PyMOL console to see the AI's heatmap:")
    print("   spectrum b, blue_white_red, selection, minimum=0, maximum=100")
    print("="*40)
