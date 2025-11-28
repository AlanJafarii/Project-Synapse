import torch
import numpy as np
from Bio.PDB import PDBParser, PDBIO

# ایمپورت ماژول‌های قبلی
from importlib import import_module
featurizer_module = import_module("02_featurizer")
model_module = import_module("03_model")

def save_predictions_to_pdb(original_pdb, output_pdb, predictions):
    """
    جایگزینی مقادیر B-factor واقعی با مقادیر پیش‌بینی شده توسط AI
    برای نمایش در PyMOL
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pred', original_pdb)
    
    # شمارنده برای پیمایش لیست پیش‌بینی‌ها
    i = 0
    
    # لیست پیش‌بینی‌ها را به فرمت ساده پایتون تبدیل میکنیم
    preds_flat = predictions.flatten().cpu().numpy()
    
    # مقیاس‌دهی دوباره (چون ما بین 0 و 1 نرمال کرده بودیم، الان ضربدر 100 میکنیم که در PyMOL بهتر دیده شه)
    preds_scaled = preds_flat * 100 
    
    for atom in structure.get_atoms():
        # همان فیلتر همیشگی
        if atom.get_parent().get_resname() == 'HOH':
            continue
            
        # ست کردن B-factor جدید (پیش‌بینی هوش مصنوعی)
        atom.set_bfactor(preds_scaled[i])
        i += 1
        
    # ذخیره فایل جدید
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"Predictions saved to: {output_pdb}")

if __name__ == "__main__":
    # 1. تنظیمات
    pdb_file = "data/pdb/pdb1crn.ent"
    output_file = "synapse_prediction.pdb"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Running Inference (Prediction Mode)...")
    
    # 2. آماده‌سازی داده
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    adj = feat.build_adjacency(pos)
    
    x = x.to(device)
    adj = adj.to(device)
    
    # 3. بازسازی مدل (باید دقیقاً همان معماری زمان آموزش باشد)
    model = model_module.SimpleGNN(input_dim=5, hidden_dim=32, output_dim=1).to(device)
    
    # نکته مهم: در پروژه واقعی، باید وزن‌های ذخیره شده (model.pth) را لود کنیم.
    # اما چون الان همه چیز در حافظه RAM هست و اسکریپت‌ها جدا هستند، 
    # ما اینجا دوباره یک مدل خام می‌سازیم و فقط یک دور سریع (Quick Train) روی آن می‌زنیم 
    # تا به حالت قبل برسد (فقط برای این دمو).
    # در پروژه نهایی، فایل `04` باید مدل را ذخیره کند و فایل `05` آن را لود کند.
    
    # --- آموزش سریع (Mini-Training) برای بازسازی هوش مدل ---
    # (چون وزن‌ها را در فایل ذخیره نکرده بودیم، اینجا سریع بازسازی میکنیم)
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    y_true = import_module("04_train").get_real_bfactors(pdb_file).to(device)
    
    print("Restoring model weights...")
    model.train()
    for _ in range(100): # 100 دور سریع
        optimizer.zero_grad()
        out = model(x, adj)
        loss = torch.nn.MSELoss()(out, y_true)
        loss.backward()
        optimizer.step()
    # -------------------------------------------------------

    # 4. پیش‌بینی نهایی
    model.eval()
    with torch.no_grad():
        predictions = model(x, adj)
    
    # 5. خروجی گرفتن
    save_predictions_to_pdb(pdb_file, output_file, predictions)
    
    print("\n" + "="*40)
    print("VISUALIZATION INSTRUCTIONS (PyMOL)")
    print("="*40)
    print("1. Open output file in PyMOL:")
    print(f"   file > open > {output_file}")
    print("2. Type this command in PyMOL console to see the AI's heatmap:")
    print("   spectrum b, blue_white_red, selection, minimum=0, maximum=100")
    print("="*40)