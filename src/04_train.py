import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Bio.PDB import PDBParser

# ایمپورت ماژول‌های قبلی خودمان
from importlib import import_module
featurizer_module = import_module("02_featurizer")
model_module = import_module("03_model")

def get_real_bfactors(pdb_path):
    """استخراج اعداد واقعی لرزش اتم‌ها (B-factor) از فایل PDB به عنوان جواب صحیح"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    bfactors = []
    
    for atom in structure.get_atoms():
        # دقیقا همان فیلترهایی که در 02_featurizer داشتیم را باید اعمال کنیم
        if atom.get_parent().get_resname() == 'HOH':
            continue
        bfactors.append(atom.get_bfactor())
    
    # نرمال‌سازی داده‌ها (بین 0 و 1) تا یادگیری راحت‌تر شود
    b_np = np.array(bfactors)
    b_norm = (b_np - b_np.min()) / (b_np.max() - b_np.min())
    
    return torch.tensor(b_norm, dtype=torch.float32).unsqueeze(1) # شکل (N, 1)

# --- بدنه اصلی آموزش ---
if __name__ == "__main__":
    # 1. آماده‌سازی داده‌ها
    pdb_file = "data/pdb/pdb1crn.ent"
    print(f"Loading data from {pdb_file}...")
    
    # ساخت گراف (ورودی مسئله)
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    adj = feat.build_adjacency(pos)
    
    # گرفتن جواب‌های صحیح (Ground Truth)
    y_true = get_real_bfactors(pdb_file)
    
    # انتقال به GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    x = x.to(device)
    adj = adj.to(device)
    y_true = y_true.to(device)
    
    # 2. ساخت مدل و ابزارهای آموزش
    model = model_module.SimpleGNN(input_dim=5, hidden_dim=32, output_dim=1).to(device)
    
    # تابع خطا (MSE): میانگین مربعات خطا (فاصله حدس تا واقعیت)
    criterion = nn.MSELoss()
    
    # بهینه‌ساز (Adam): مسئول تغییر وزن‌های مغز
    # lr=0.01 سرعت یادگیری است
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("\n" + "="*30)
    print("STARTING TRAINING LOOP")
    print("="*30)
    
    # 3. حلقه آموزش (Training Loop)
    epochs = 200 # 200 بار تمرین کن
    
    for epoch in range(epochs):
        model.train() # مدل را در حالت یادگیری قرار بده
        
        # الف) پاک کردن حافظه گرادیان‌های قبلی
        optimizer.zero_grad()
        
        # ب) پیش‌بینی (Forward Pass)
        y_pred = model(x, adj)
        
        # ج) محاسبه خطا (Loss)
        loss = criterion(y_pred, y_true)
        
        # د) یادگیری (Backward Pass)
        loss.backward() # محاسبه اینکه چقدر باید تغییر کنیم
        optimizer.step() # اعمال تغییرات
        
        # چاپ وضعیت هر 20 دور یکبار
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.6f}")

    print("="*30)
    print("TRAINING COMPLETED")
    
    # 4. تست نهایی
    model.eval()
    with torch.no_grad():
        final_pred = model(x, adj)
        
    print("\nSample Results (Normalized B-factors):")
    print(f"{'Atom Idx':<10} | {'Prediction':<10} | {'Actual (True)':<10}")
    print("-" * 35)
    # نمایش 5 اتم اول و 5 اتم آخر برای مقایسه
    indices = [0, 1, 2, 100, 101, 102] 
    for i in indices:
        print(f"{i:<10} | {final_pred[i].item():.4f}     | {y_true[i].item():.4f}")