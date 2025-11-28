import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import os

class SynapseCore:
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        print(f"Loading Pre-trained Model: {model_name}...")
       self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        print("Model Loaded Successfully!")

    def predict_property(self, smiles_list):
        """پیش‌بینی خواص دارو با استفاده از دانش ChemBERTa"""
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=1)
        return probs

    def save_my_model(self, save_directory="./synapse_v1_weights"):
       
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        print(f"Saving Synapse Brain to {save_directory}...")
        
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        print("✅ Saved! You can now send this folder to anyone.")

if __name__ == "__main__":
    synapse = SynapseCore()
    
    drugs = [
        "CC(=O)Oc1ccccc1C(=O)O", # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "CCO" # Ethanol
    ]
    
    print("\nRunning Inference with ChemBERTa...")
    predictions = synapse.predict_property(drugs)
    
    print("-" * 40)
    for i, drug in enumerate(drugs):
        score = predictions[i][1].item() * 100 
        print(f"Drug: {drug[:15]}... | Activity Score: {score:.2f}%")
    print("-" * 40)
    
    synapse.save_my_model()
