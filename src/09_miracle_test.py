import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class SynapseBrain:
    def __init__(self):
        print("Waking up the AI Chemist...")
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print("Brain is ready!")

    def get_embedding(self, smile):
        inputs = self.tokenizer(smile, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
  
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding

if __name__ == "__main__":
    brain = SynapseBrain()
    
    target_name = "Aspirin"
    target_smile = "CC(=O)Oc1ccccc1C(=O)O"
    
    print(f"\nTarget Molecule: {target_name}")
    print(f"Formula: {target_smile}")
    print("-" * 50)
    
    target_vec = brain.get_embedding(target_smile)

    candidates = [
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),  # مسکن
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),        # مسکن 
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"), # محرک 
        ("Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O"),      # قند 
        ("Water", "O"),                               # آب
    ]

    print(f"{'Candidate':<15} | {'Similarity Score (0-100%)':<25} | {'AI Opinion'}")
    print("-" * 55)

    results = []
    
    for name, smile in candidates:
        candidate_vec = brain.get_embedding(smile)
        
        similarity = F.cosine_similarity(target_vec, candidate_vec)
        score = similarity.item() * 100
        
        results.append((name, score))

    results.sort(key=lambda x: x[1], reverse=True)

    for name, score in results:
        comment = ""
        if score > 95: comment = "IT'S A CLONE!"
        elif score > 85: comment = "Very Similar (Family)"
        elif score > 75: comment = "Somewhat Related"
        else: comment = "Totally Different"
        
        print(f"{name:<15} | {score:.2f}%{' '*19} | {comment}")
    
    print("-" * 55)
    print("ANALYSIS:")
    print("If Ibuprofen or Paracetamol are at the top, the AI understands 'Painkillers'.")
    print("If Glucose/Water are at the bottom, the AI understands structural differences.")
