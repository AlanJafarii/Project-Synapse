import torch
import numpy as np
from Bio.PDB import PDBParser
import os

# دیکشنری برای تبدیل نام اتم به عدد (Encoding)
# C=0, N=1, O=2, S=3, Others=4
ATOM_MAP = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

class ProteinFeaturizer:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def get_atom_features(self, element):
        """تبدیل عنصر به یک بردار وان-هات (One-Hot Vector)"""
        # مثلا کربن میشه: [1, 0, 0, 0, 0]
        # نیتروژن میشه: [0, 1, 0, 0, 0]
        idx = ATOM_MAP.get(element, 4) # اگر ناشناخته بود بذار 4
        one_hot = np.zeros(5)
        one_hot[idx] = 1.0
        return one_hot

    def process_pdb(self, pdb_path):
        """تبدیل فایل PDB به تانسورهای پای‌تورچ"""
        structure = self.parser.get_structure('protein', pdb_path)
        
        node_features = []  # ویژگی‌های هر اتم (جنس اتم)
        coords = []         # مختصات (x, y, z)

        # فقط اتم‌های استاندارد پروتئین را برمیداریم (آب را حذف میکنیم)
        for atom in structure.get_atoms():
            # فیلتر کردن مولکول‌های آب (HOH)
            if atom.get_parent().get_resname() == 'HOH':
                continue
                
            # 1. گرفتن مختصات
            coords.append(atom.get_coord())
            
            # 2. گرفتن جنس اتم (C, N, O, ...)
            element = atom.element
            if not element: # گاهی اوقات فایل PDB عنصر را ندارد، از نام حدس میزنیم
                element = atom.get_name()[0]
            
            node_features.append(self.get_atom_features(element))

        # تبدیل به تانسور پای‌تورچ (زبان هوش مصنوعی)         # X: ویژگی‌های اتم‌ها (N_atoms, 5)
        X = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # Pos: مختصات (N_atoms, 3)
        Pos = torch.tensor(np.array(coords), dtype=torch.float32)
        
        return X, Pos

    def build_adjacency(self, coords, threshold=4.5):
        """ساخت ماتریس همسایگی بر اساس فاصله (هر کی زیر 4.5 آنگستروم بود همسایه است)"""
        # محاسبه فاصله همه با همه (Pairwise Distance)
        dist_matrix = torch.cdist(coords, coords)
        
        # ماسک کردن: جاهایی که فاصله کمتر از حد مجاز است و خود اتم نیست
        adj_matrix = (dist_matrix < threshold) & (dist_matrix > 0)
        
        return adj_matrix.float()

# --- تست کد ---
if __name__ == "__main__":
    # آدرس فایلی که مرحله قبل دانلود کردی
    pdb_file = "data/pdb/pdb1crn.ent" 
    
    if os.path.exists(pdb_file):
        featurizer = ProteinFeaturizer()
        
        print("Processing Protein Graph...")
        
        # 1. تبدیل اتم‌ها به ویژگی و مختصات
        node_feats, coordinates = featurizer.process_pdb(pdb_file)
        
        # 2. پیدا کردن همسایه‌ها (ساخت گراف)
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
        
        # انتقال به GPU اگر موجود باشد (برای تست CUDA)
        if torch.cuda.is_available():
            print("\nCUDA Check: Moving tensors to GPU...")
            node_feats = node_feats.to('cuda')
            print("Success! Tensors are now on NVIDIA GPU.")
        else:
            print("\nWarning: CUDA not available. Running on CPU.")
            
    else:
        print("Error: PDB file not found. Run 01_data_loader.py first.")