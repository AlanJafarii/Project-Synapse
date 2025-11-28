import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser # فقط برای لود کردن دیتای تست

# ایمپورت کردن کدهای مراحل قبل (برای اینکه دوباره ننویسیم)
from importlib import import_module
featurizer_module = import_module("02_featurizer") # ترفند ایمپورت فایل‌های عددی

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        
        # لایه اول: پیام‌رسانی (Message Passing)
        # این ماتریس وزن W1 یاد میگیره چطور ویژگی‌های اتم رو ترکیب کنه
        self.W1 = nn.Linear(input_dim, hidden_dim)
        
        # لایه دوم: تصمیم‌گیری نهایی
        self.W2 = nn.Linear(hidden_dim, output_dim)
        
        print(f"Model Initialized: Input={input_dim} -> Hidden={hidden_dim} -> Output={output_dim}")

    def forward(self, x, adj):
        """
        x: ویژگی‌های اتم‌ها (N, 5)
        adj: ماتریس همسایگی (N, N)
        """
        # 1. تغییر شکل ویژگی‌ها (Feature Transformation)
        # فرمول: H = X * W1
        h = self.W1(x) 
        
        # 2. تبادل پیام (Message Passing / Aggregation)
        # فرمول: H_new = Adjacency * H
        # این یعنی هر اتم، ویژگی‌های همسایه‌هاش رو جمع میزنه
        h = torch.mm(adj, h)
        
        # 3. تابع فعال‌ساز (Activation) - مثل نورون واقعی
        h = F.relu(h)
        
        # 4. لایه دوم (خروجی)
        # مثلاً پیش‌بینی اینکه این اتم چقدر حرکت میکنه (B-factor)
        out = self.W2(h)
        
        return out

# --- تست کد ---
if __name__ == "__main__":
    # 1. آماده‌سازی داده (مثل مرحله قبل)
    pdb_file = "data/pdb/pdb1crn.ent"
    feat = featurizer_module.ProteinFeaturizer()
    x, pos = feat.process_pdb(pdb_file)
    adj = feat.build_adjacency(pos)
    
    # انتقال به GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    adj = adj.to(device)
    
    print("\n" + "="*30)
    print("INITIALIZING SYNAPSE AI ENGINE")
    print("="*30)
    
    # 2. ساخت مدل
    # ورودی: 5 (چون اتم‌ها رو One-Hot کردیم: C, N, O, S, Other)
    # مخفی: 16 (تعداد نورون‌های لایه وسط)
    # خروجی: 1 (مثلاً پیش‌بینی یک خاصیت برای هر اتم)
    model = SimpleGNN(input_dim=5, hidden_dim=16, output_dim=1).to(device)
    
    # 3. اجرای مدل (Forward Pass)
    # اینجا جادو اتفاق میافته!
    with torch.no_grad(): # چون فعلا آموزش نمیدیم، گرادیان نمیخوایم
        prediction = model(x, adj)
    
    print("\n--- Model Output Analysis ---")
    print(f"Input Shape: {x.shape} (Atoms, Features)")
    print(f"Output Shape: {prediction.shape} (Atoms, Predicted_Value)")
    
    # نمایش پیش‌بینی برای 5 اتم اول
    print("\nFirst 5 Atoms Predictions (Random Weights):")
    print(prediction[:5].cpu().numpy())
    
    print("\nStatus: The Brain is ALIVE and processing geometry!")