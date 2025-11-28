import os
from Bio.PDB import PDBList, PDBParser
import numpy as np
import warnings

# جلوگیری از هشدارهای غیرمهم بایوپایتون
warnings.filterwarnings('ignore')

class ProteinLoader:
    def __init__(self):
        self.pdbl = PDBList()
        self.parser = PDBParser()

    def download_and_parse(self, pdb_id, save_dir="data/pdb"):
        """
        این تابع یک شناسه PDB میگیرد، فایل را دانلود میکند و اطلاعات اتم‌های آن را استخراج میکند.
        """
        # 1. ساخت دایرکتوری اگر وجود ندارد
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"Downloading {pdb_id}...")
        # دانلود فایل PDB از سرور جهانی
        file_path = self.pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format="pdb")
        
        # 2. پارس کردن ساختار
        structure = self.parser.get_structure(pdb_id, file_path)
        
        # 3. استخراج مختصات (Coordinates)
        coords = []
        atoms_count = 0
        
        # پیمایش مدل -> زنجیره -> رزیدیو -> اتم
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords.append(atom.get_coord())
                        atoms_count += 1
        
        coords_np = np.array(coords)
        
        return coords_np, file_path

# --- تست کد ---
if __name__ == "__main__":
    loader = ProteinLoader()
    
    # تست با یک پروتئین کوچک (1CRN: پروتئین کرامبین - کلاسیک برای تست)
    target_pdb = "1CRN"
    
    try:
        atom_coords, path = loader.download_and_parse(target_pdb)
        
        print("\n" + "="*30)
        print(f"Project Synapse - Data Module")
        print("="*30)
        print(f"File saved at: {path}")
        print(f"Total Atoms: {atom_coords.shape[0]}")
        print(f"Coordinate Shape: {atom_coords.shape}") # باید (N, 3) باشد
        print(f"Center of Mass (Geometric): {np.mean(atom_coords, axis=0)}")
        print("="*30)
        
        print("\nNext Step: Open the downloaded .ent file in PyMOL to verify!")
        
    except Exception as e:
        print(f"Error: {e}")