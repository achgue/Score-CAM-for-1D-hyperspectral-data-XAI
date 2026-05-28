import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURAZIONE ---
# Sostituisci con il percorso reale del tuo file .pt
FILE_PATH = 'output_pt/baumlein/model_train_split/test/Papier_Leinen_712Z_000000Norm_test.pt'  
SAMPLE_IDX = 0  # Quale campione vuoi vedere (es. il primo, il decimo...)

# --- FUNZIONI DI PREPROCESSING (Per vedere come appare alla rete) ---
def compute_snv(tensor_1d):
    """Standard Normal Variate su un singolo spettro 1D"""
    mean = tensor_1d.mean()
    std = tensor_1d.std()
    return (tensor_1d - mean) / (std + 1e-8)

def rescale_zero_one(tensor_1d):
    """Min-Max scaling su un singolo spettro 1D"""
    return (tensor_1d - tensor_1d.min()) / (tensor_1d.max() - tensor_1d.min() + 1e-8)

def inspect_and_plot(file_path, sample_idx):
    if not os.path.exists(file_path):
        print(f"ERRORE: Il file {file_path} non esiste.")
        return

    print(f"--- Caricamento file: {file_path} ---")
    
    # 1. Caricamento Tensor
    # Nota: torch.load carica tutto in RAM. Se il file è enorme (GBs), potrebbe essere lento.
    data = torch.load(file_path, map_location='cpu')
    
    # Gestione caso in cui il .pt sia un oggetto Dataset o una tupla (Data, Labels)
    if isinstance(data, (tuple, list)):
        print("Il file contiene una tupla (Dati, Label). Seleziono i Dati [0].")
        data = data[0]
    elif hasattr(data, 'tensors'):
        print("Il file è un TensorDataset. Estraggo i tensor interni.")
        data = data.tensors[0]

    print(f"Shape totale del dataset: {data.shape}")
    
    # Controllo indice
    if sample_idx >= data.shape[0]:
        print(f"ERRORE: sample_idx {sample_idx} è fuori range (Max: {data.shape[0]-1})")
        return

    # 2. Estrazione Campione
    # Assumiamo forma [N, Channels, H, W] -> es. [2000, 384, 3, 3]
    raw_sample = data[sample_idx]
    print(f"Shape del singolo campione: {raw_sample.shape}")

    # Estraiamo il pixel centrale (1, 1) per avere lo spettro 1D
    # Se il dato è già 1D [Channels], lo usiamo direttamente
    if raw_sample.dim() == 3: # [Channels, H, W]
        spectrum_raw = raw_sample[:, 1, 1]
        print("Estratto pixel centrale (1,1) dalla patch 3x3.")
    elif raw_sample.dim() == 1: # [Channels]
        spectrum_raw = raw_sample
        print("Il dato era già 1D.")
    else:
        print(f"Formato non standard: {raw_sample.dim()} dimensioni. Provo appiattimento.")
        spectrum_raw = raw_sample.flatten()

    # Conversione in Numpy per il plot
    spectrum_raw_np = spectrum_raw.numpy()

    # 3. Preprocessing (Simulazione Pipeline)
    spectrum_snv = compute_snv(spectrum_raw)
    spectrum_final = rescale_zero_one(spectrum_snv).numpy()

    # 4. Plotting
    plt.figure(figsize=(12, 6))

    # Plot Raw
    plt.subplot(1, 2, 1)
    plt.plot(spectrum_raw_np, color='tab:blue')
    plt.title(f"Spettro Grezzo (Sample {sample_idx})")
    plt.xlabel("Banda")
    plt.ylabel("Intensità")
    plt.grid(True, alpha=0.3)

    # Plot Preprocessed
    plt.subplot(1, 2, 2)
    plt.plot(spectrum_final, color='tab:orange')
    plt.title(f"Spettro Preprocessato (SNV + 0-1)")
    plt.xlabel("Banda")
    plt.ylabel("Normalized Intensity")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Salva o mostra
    out_file = "test_plot_sample.png"
    plt.savefig(out_file)
    print(f"\nGrafico salvato in: {out_file}")
    # plt.show() # Decommenta se hai un monitor

if __name__ == "__main__":
    inspect_and_plot(FILE_PATH, SAMPLE_IDX)