import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ایمپورت ماژول‌های قبلی
from importlib import import_module
featurizer_module = import_module("02_featurizer")

class InteractionModel(nn.Module):
    def __init__(self, protein_dim, drug_dim):
        super(InteractionModel, self).__init__()
        
        # شاخه اول: پردازش پروتئین
        self.protein_net = nn.Sequential(
            nn.Linear(protein_dim, 32),
            nn.ReLU()
        )
        
        # شاخه دوم: پردازش دارو
        self.drug_net = nn.Sequential(
            nn.Linear(drug_dim, 32),
            nn.ReLU()
        )
        
        # شاخه سوم: ترکیب (Fusion Core)
        # 32 (protein) + 32 (drug) = 64 ورودی
        self.fusion_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1), # خروجی: امتیاز اتصال (Binding Score)
            nn.Sigmoid() # خروجی بین 0 تا 1 (0: عدم اتصال، 1: اتصال قوی)
        )
        
    def forward(self, protein_vec, drug_vec):
        p = self.protein_net(protein_vec)
        d = self.drug_net(drug_vec)
        
        # چسباندن دو بردار به هم (Concatenation)
        combined = torch.cat((p, d), dim=1)
        
        score = self.fusion_net(combined)
        return score

def get_drug_fingerprint(smiles):
    """تبدیل رشته متنی دارو به اثرانگشت شیمیایی (Morgan Fingerprint)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # ایجاد یک بردار بیت (0 و 1) که ساختار دارو را نشان میدهد
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
    return torch.tensor(np.array(fp), dtype=torch.float32).unsqueeze(0) # (1, 128)

def get_protein_embedding(pdb_file, device):
    """گرفتن میانگین ویژگی‌های گراف پروتئین"""
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    # میانگین گرفتن از تمام اتم‌ها برای بدست آوردن یک بردار کلی برای کل پروتئین
    # (در مدل‌های پیشرفته‌تر از Graph Pooling استفاده میشود)
    protein_vec = torch.mean(x, dim=0).unsqueeze(0).to(device) # (1, 5)
    return protein_vec

# --- بدنه اصلی ---
if __name__ == "__main__":
    print("Welcome to Project Synapse Virtual Lab")
    print("="*40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb_file = "data/pdb/pdb1crn.ent" # پروتئین هدف: Crambin
    
    # 1. لود کردن پروتئین
    print(f"Target Protein: Crambin (1CRN)")
    try:
        p_vec = get_protein_embedding(pdb_file, device)
        print("Protein Encoded Successfully.")
    except Exception as e:
        print("Run 01_data_loader.py first!")
        exit()

    # 2. ساخت مدل اتصال (Interaction Model)
    # ورودی پروتئین: 5 (تعداد اتم‌های وان-هات)
    # ورودی دارو: 128 (سایز فینگرپرینت RDKit)
    model = InteractionModel(protein_dim=5, drug_dim=128).to(device)
    model.eval() # مدل در حالت تست (با وزن‌های تصادفی چون آموزش ندادیم)
    
    print("\nAttempting to dock molecules generated in Step 6...")
    print("-" * 50)
    print(f"{'Drug Name/SMILES':<30} | {'Synapse Score (0-100%)':<20} | {'Status'}")
    print("-" * 50)
    
    # لیست داروها (شامل تولیدات خودمان و چند داروی سمی برای تست)
    test_drugs = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ethanol", "CCO"),
        ("Vanillin", "COc1cc(C=O)ccc1O"),
        ("Toxic Cyanide", "C#N"), # سیانور (باید نمره متفاوتی بگیرد)
        ("Complex Drug", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C") # کافئین
    ]
    
    for name, smiles in test_drugs:
        # تبدیل دارو به بردار
        d_vec = get_drug_fingerprint(smiles)
        
        if d_vec is not None:
            d_vec = d_vec.to(device)
            
            # پیش‌بینی هوش مصنوعی
            with torch.no_grad():
                prediction = model(p_vec, d_vec)
                
            score = prediction.item() * 100
            
            # تفسیر نمره (چون مدل آموزش ندیده، این فقط دمو معماری است)
            status = "Strong Binder" if score > 50 else "Weak Binder"
            
            print(f"{name:<30} | {score:.2f}%               | {status}")
        else:
            print(f"{name:<30} | INVALID SMILES")
            
    print("-" * 50)
    print("Note: Scores are based on random weights (Untrained Model).")
    print("In a real scenario, we would train this on the ChEMBL database.")
    print("="*40)