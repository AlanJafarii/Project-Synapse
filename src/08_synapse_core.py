import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import os

class SynapseCore:
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        print(f"Loading Pre-trained Model: {model_name}...")
        # 1. دانلود توکنایزر (مترجم زبان شیمی)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 2. دانلود مغز مدل (Weights)
        # ما اینجا مدل را برای کلاسیفیکیشن (آیا دارو موثر است؟) تنظیم میکنیم
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        print("Model Loaded Successfully!")

    def predict_property(self, smiles_list):
        """پیش‌بینی خواص دارو با استفاده از دانش ChemBERTa"""
        # تبدیل متن داروها به فرمت قابل فهم برای مدل
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # تبدیل خروجی به احتمال (Probability)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs

    def save_my_model(self, save_directory="./synapse_v1_weights"):
        """
        این همان تابعی است که مشکل قبلی تو را حل میکند!
        ذخیره مدل روی هارد دیسک برای اشتراک‌گذاری یا استفاده بعدی.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        print(f"Saving Synapse Brain to {save_directory}...")
        
        # ذخیره مدل و توکنایزر
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        print("✅ Saved! You can now send this folder to anyone.")

# --- اجرای تست ---
if __name__ == "__main__":
    # 1. ساخت هسته ساینپس
    synapse = SynapseCore()
    
    # 2. تست روی چند دارو
    drugs = [
        "CC(=O)Oc1ccccc1C(=O)O", # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "CCO" # Ethanol
    ]
    
    print("\nRunning Inference with ChemBERTa...")
    predictions = synapse.predict_property(drugs)
    
    print("-" * 40)
    for i, drug in enumerate(drugs):
        # چون مدل هنوز روی دیتای خاص ما فاین‌تیون نشده، این اعداد دانش عمومی مدل هستند
        score = predictions[i][1].item() * 100 
        print(f"Drug: {drug[:15]}... | Activity Score: {score:.2f}%")
    print("-" * 40)
    
    # 3. ذخیره ابدی!
    # اینجاست که فایل‌هایی که دانشجو میخواست تولید میشوند
    synapse.save_my_model()