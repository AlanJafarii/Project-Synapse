import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from importlib import import_module
featurizer_module = import_module("02_featurizer")

class InteractionModel(nn.Module):
    def __init__(self, protein_dim, drug_dim):
        super(InteractionModel, self).__init__()
        
        self.protein_net = nn.Sequential(
            nn.Linear(protein_dim, 32),
            nn.ReLU()
        )
        
        self.drug_net = nn.Sequential(
            nn.Linear(drug_dim, 32),
            nn.ReLU()
        )
        
        self.fusion_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, protein_vec, drug_vec):
        p = self.protein_net(protein_vec)
        d = self.drug_net(drug_vec)
        
        combined = torch.cat((p, d), dim=1)
        
        score = self.fusion_net(combined)
        return score

def get_drug_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
    return torch.tensor(np.array(fp), dtype=torch.float32).unsqueeze(0) 

def get_protein_embedding(pdb_file, device):
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    protein_vec = torch.mean(x, dim=0).unsqueeze(0).to(device) 
    return protein_vec

if __name__ == "__main__":
    print("Welcome to Project Synapse Virtual Lab")
    print("="*40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb_file = "data/pdb/pdb1crn.ent" 
    
    print(f"Target Protein: Crambin (1CRN)")
    try:
        p_vec = get_protein_embedding(pdb_file, device)
        print("Protein Encoded Successfully.")
    except Exception as e:
        print("Run 01_data_loader.py first!")
        exit()

 
    model = InteractionModel(protein_dim=5, drug_dim=128).to(device)
    model.eval() 
    
    print("\nAttempting to dock molecules generated in Step 6...")
    print("-" * 50)
    print(f"{'Drug Name/SMILES':<30} | {'Synapse Score (0-100%)':<20} | {'Status'}")
    print("-" * 50)
    
    test_drugs = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ethanol", "CCO"),
        ("Vanillin", "COc1cc(C=O)ccc1O"),
        ("Toxic Cyanide", "C#N"), # سیانور
        ("Complex Drug", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C") # کافئین
    ]
    
    for name, smiles in test_drugs:
        d_vec = get_drug_fingerprint(smiles)
        
        if d_vec is not None:
            d_vec = d_vec.to(device)
            
            with torch.no_grad():
                prediction = model(p_vec, d_vec)
                
            score = prediction.item() * 100
            
            status = "Strong Binder" if score > 50 else "Weak Binder"
            
            print(f"{name:<30} | {score:.2f}%               | {status}")
        else:
            print(f"{name:<30} | INVALID SMILES")
            
    print("-" * 50)
    print("Note: Scores are based on random weights (Untrained Model).")
    print("In a real scenario, we would train this on the ChEMBL database.")
    print("="*40)
