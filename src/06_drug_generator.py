import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rdkit import Chem

# لیست کوچکی از مولکول‌های دارویی معروف (SMILES) برای آموزش
DRUG_CORPUS = [
    "CC(=O)Oc1ccccc1C(=O)O", # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
    "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C", # Penicillin G
    "CN(C)C(=N)NC(=N)N", # Metformin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", # Ibuprofen
    "COc1cc(C=O)ccc1O", # Vanillin
    "CC(=O)NO", # Acetohydroxamic acid
    "C1ccccc1", # Benzene (Base)
    "CCO", # Ethanol
]

class ChemicalLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super(ChemicalLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # لایه امبدینگ: تبدیل کاراکتر به بردار
        self.embedding = nn.Embedding(vocab_size, 32)
        
        # لایه LSTM: مغز مدل که حافظه دارد
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        
        # لایه خروجی: پیش‌بینی کاراکتر بعدی
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        # x: (batch, seq_len)
        embeds = self.embedding(x)
        # out: (batch, seq_len, hidden_dim)
        out, hidden = self.lstm(embeds, hidden)
        # reshape برای لایه خطی
        out = out.reshape(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

# --- توابع کمکی ---
def get_vocab(molecules):
    chars = set(''.join(molecules))
    chars.add('<EOS>') # توکن ویژه پایان
    vocab_to_int = {c: i for i, c in enumerate(sorted(chars))}
    int_to_vocab = {i: c for c, i in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab

# --- بدنه اصلی ---
if __name__ == "__main__":
    print("Initializing Generative Chemistry Module...")
    
    # 1. پردازش متن (Tokenization)
    vocab_to_int, int_to_vocab = get_vocab(DRUG_CORPUS)
    vocab_size = len(vocab_to_int)
    print(f"Vocabulary Size: {vocab_size} unique tokens")
    
    # گرفتن کد عددی مربوط به <EOS>
    eos_id = vocab_to_int['<EOS>']

    # تنظیمات مدل
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChemicalLanguageModel(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(DRUG_CORPUS)} molecules...")
    
    # 2. حلقه آموزش
    for epoch in range(200):
        total_loss = 0
        
        for smi in DRUG_CORPUS:
            # --- بخش اصلاح شده ---
            # اول حروف را به عدد تبدیل میکنیم
            encoded = [vocab_to_int[c] for c in smi]
            # سپس کد پایان را به لیست اضافه میکنیم
            encoded.append(eos_id)
            # ---------------------
            
            # تبدیل به تانسور
            input_seq = torch.tensor(encoded[:-1], dtype=torch.long).unsqueeze(0).to(device)
            target_seq = torch.tensor(encoded[1:], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            # اجرا و محاسبه خطا
            output, _ = model(input_seq, None)
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(DRUG_CORPUS):.4f}")

    print("\nTRAINING COMPLETE. Generating new molecules...")
    
    # 3. تولید مولکول جدید (Sampling)
    model.eval()
    
    def generate_molecule(start_char='C', max_len=30):
        if start_char not in vocab_to_int:
            return "Error: Start char not in vocab"
            
        current_idx = vocab_to_int[start_char]
        mol_str = start_char
        hidden = None
        
        # ورودی اولیه
        input_token = torch.tensor([[current_idx]], dtype=torch.long).to(device)
        
        for _ in range(max_len):
            out, hidden = model(input_token, hidden)
            
            # تبدیل خروجی به احتمال
            probs = torch.softmax(out, dim=1).cpu().detach().numpy().flatten()
            
            # انتخاب کاراکتر بعدی
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = int_to_vocab[next_idx]
            
            if next_char == '<EOS>':
                break
                
            mol_str += next_char
            
            # آماده‌سازی ورودی بعدی
            input_token = torch.tensor([[next_idx]], dtype=torch.long).to(device)
            
        return mol_str

    print("-" * 30)
    valid_mols = 0
    for i in range(5):
        # تلاش برای تولید با حرف C (کربن) شروع میشود
        new_smiles = generate_molecule(start_char='C')
        print(f"Generated #{i+1}: {new_smiles}")
        
        # اعتبارسنجی با RDKit
        mol = Chem.MolFromSmiles(new_smiles)
        if mol:
            print("  ✅ Valid Molecule!")
            valid_mols += 1
        else:
            print("  ❌ Invalid Syntax")
            
    print("-" * 30)
    print(f"Success Rate: {valid_mols}/5")