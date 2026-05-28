import os
import glob
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, split_ratios=(0.7, 0.15, 0.15), seed=42):
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
        
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        print(f"\n--- Splitting: {folder_name} ---")
        
        # Troviamo tutti i prefissi unici (es. Metal_Immagine1 e Background_Immagine1)
        pt_files = glob.glob(os.path.join(folder, "*_part*.pt"))
        if not pt_files: continue
            
        prefixes = set([os.path.basename(f).split('_part')[0] for f in pt_files])
        
        for prefix in prefixes:
            # Carica tutte le parti per questa specifica classe/immagine
            class_files = glob.glob(os.path.join(folder, f"{prefix}_part*.pt"))
            
            all_tensors = []
            for f in class_files:
                all_tensors.append(torch.load(f))
                
            full_data = torch.cat(all_tensors, dim=0)
            total_samples = full_data.shape[0]
            
            indices = np.arange(total_samples)
            
            # Split train vs val+test
            train_idx, temp_idx = train_test_split(
                indices, train_size=split_ratios[0], random_state=seed, shuffle=True
            )
            
            relative_val_size = split_ratios[1] / (split_ratios[1] + split_ratios[2])
            val_idx, test_idx = train_test_split(
                temp_idx, train_size=relative_val_size, random_state=seed, shuffle=True
            )
            
            # Salvataggio nelle cartelle finali
            splits = {
                'train': full_data[train_idx],
                'val':   full_data[val_idx],
                'test':  full_data[test_idx]
            }
            
            for split_name, tensor_data in splits.items():
                if len(tensor_data) > 0:
                    # Formato finale: Metal_Immagine1_train.pt
                    save_name = f"{prefix}_{split_name}.pt"
                    save_path = os.path.join(output_dir, split_name, save_name)
                    
                    torch.save(tensor_data, save_path)
                    print(f"   -> {split_name.upper()}: salvato {save_name} ({len(tensor_data)} samples)")
            
            del full_data, all_tensors