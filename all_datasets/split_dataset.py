import os
import glob
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, split_ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Divide i file .pt (patch) in Train, Val e Test.
    
    Args:
        input_dir (str): Cartella contenente le sottocartelle con i file .pt (es. dataset_baupol_patches)
        output_dir (str): Cartella dove salvare gli split
        split_ratios (tuple): (train_%, val_%, test_%) - la somma deve fare 1.0
        seed (int): Per riproducibilità
    """
    
    assert sum(split_ratios) == 1.0, "Le percentuali devono sommare a 1.0"
    
    # 1. Configurazione Cartelle Output
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
        
    # 2. Trova le sottocartelle (ogni sottocartella è un'immagine/classe originale)
    # Esempio: dataset_baupol_patches/Matera_ROI1/
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    
    print(f"Trovate {len(subfolders)} sorgenti dati (ROI/Immagini). Inizio split...")

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\n--- Processing: {folder_name} ---")
        
        # 3. Carica TUTTI i chunk di questa cartella per unirli e mischiarli
        pt_files = glob.glob(os.path.join(folder, "*_part*.pt"))
        
        if not pt_files:
            print(f"   [SKIP] Nessun file patch trovato in {folder_name}")
            continue
            
        # Carichiamo tutti i tensori in memoria (si assume che un singolo ROI entri in RAM)
        # Se non entrano, bisognerebbe fare uno split "per file" meno preciso.
        all_tensors = []
        total_samples = 0
        
        try:
            for f in pt_files:
                t = torch.load(f)
                all_tensors.append(t)
                total_samples += t.shape[0]
            
            # Unione in un unico tensore temporaneo
            full_data = torch.cat(all_tensors, dim=0) # Shape: (Total_N, C, H, W)
            print(f"   Totale campioni: {total_samples}")
            
            # --- 4. CALCOLO INDICI SPLIT ---
            indices = np.arange(total_samples)
            
            # Primo split: Train vs (Val + Test)
            train_idx, temp_idx = train_test_split(
                indices, 
                train_size=split_ratios[0], 
                random_state=seed,
                shuffle=True
            )
            
            # Secondo split: Val vs Test
            # Ricalcoliamo la percentuale relativa per il secondo split
            relative_val_size = split_ratios[1] / (split_ratios[1] + split_ratios[2])
            
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=relative_val_size,
                random_state=seed,
                shuffle=True
            )
            
            # --- 5. SALVATAGGIO NEI RISPETTIVI FOLDER ---
            splits = {
                'train': full_data[train_idx],
                'val': full_data[val_idx],
                'test': full_data[test_idx]
            }
            
            for split_name, tensor_data in splits.items():
                if len(tensor_data) > 0:
                    # Nome file: Matera_ROI1_train.pt
                    save_name = f"{folder_name}_{split_name}.pt"
                    save_path = os.path.join(output_dir, split_name, save_name)
                    
                    torch.save(tensor_data, save_path)
                    print(f"   -> {split_name.upper()}: salvati {len(tensor_data)} campioni in {save_name}")
            
            # Pulizia RAM
            del full_data, all_tensors, splits
            
        except Exception as e:
            print(f"   [ERRORE] Impossibile splittare {folder_name}: {e}")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    INPUT_PATCHES_DIR = "./dataset_baupol_patches" # La cartella creata dallo script precedente
    OUTPUT_SPLIT_DIR = "./dataset_baupol_split"    # Dove finiranno i dati pronti
    
    # Percentuali: 70% Train, 15% Validation, 15% Test
    RATIOS = (0.70, 0.15, 0.15) 
    
    split_dataset(INPUT_PATCHES_DIR, OUTPUT_SPLIT_DIR, RATIOS)