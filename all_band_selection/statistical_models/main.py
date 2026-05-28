import numpy as np
import os
#import matplotlib.pyplot as plt
from src.utils import load_data_matrix
from configuration import DATASET_CATALOG, MODELS, SELECTION_PARAMS, PREPROCESSING_CONFIG

def save_bands_to_txt(indices, filepath):
    """Salva una lista di indici interi in un file txt."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for idx in indices:
            f.write(f"{int(idx)}\n")
    print(f"   [Saved] Bands saved to: {filepath}")

def run_comparison():
    target_bands = SELECTION_PARAMS['n_output_bands']
    output_dir = SELECTION_PARAMS['output_dir']
    
    # --- CICLO SUI DATASET ---
    for ds_conf in DATASET_CATALOG:
        ds_name = ds_conf['name']
        print(f"\n{'='*50}")
        print(f"PROCESSING DATASET: {ds_name}")
        print(f"{'='*50}")
        
        # 1. Caricamento Dati (Una volta per dataset)
        try:
            X, y = load_data_matrix(
                ds_conf['path'], 
                ds_conf['class_map'], 
                trim_start=PREPROCESSING_CONFIG['trim_start'], 
                trim_end=PREPROCESSING_CONFIG['trim_end'],
                max_samples=SELECTION_PARAMS['max_samples_per_class']
            )
        except Exception as e:
            print(f"Skipping {ds_name} due to data error: {e}")
            continue
            
        # --- CICLO SUI METODI ---
        for method_name, method_conf in MODELS.items():
            print(f"\n   --- Running Method: {method_name} ---")
            
            select_func = method_conf['func']
            extra_params = method_conf['params']
            
            # Costruiamo il nome file di output: configs/selected_bands/bands_baumlein_SPA.txt
            filename = f"bands_{ds_name}_{method_name}.txt"
            save_path = os.path.join(output_dir, filename)
            
            # Se il file esiste già, potresti volerlo saltare o sovrascrivere
            # if os.path.exists(save_path): print("Skipping..."); continue
            
            try:
                # 2. Esecuzione Algoritmo di Selezione
                # Assumiamo che la tua firma sia: select_bands(X, y, n_bands=..., **kwargs)
                # Restituisce: (selected_indices, importance_scores_opzionale) o solo indices
                
                result = select_func(
                    X, y, 
                    n_bands=target_bands, 
                    plot=False, # Gestiamo i plot esternamente se vogliamo
                    **extra_params
                )
                
                # Gestione se la funzione restituisce una tupla (indices, scores) o solo indices
                if isinstance(result, tuple):
                    selected_indices = result[0]
                else:
                    selected_indices = result
                
                # Ordiniamo gli indici (spesso aiuta averli in ordine crescente spettrale)
                #selected_indices = np.sort(np.array(selected_indices))
                selected_indices = np.array(selected_indices)
                

                # L'algoritmo vede l'indice 0, ma per noi è 0 + trim_start
                original_indices = selected_indices + PREPROCESSING_CONFIG['trim_start']

                print(f"   Selected {len(original_indices)} bands: {original_indices}")
                
                # 3. Salvataggio
                save_bands_to_txt(original_indices, save_path)
                
            except Exception as e:
                print(f"   [Error] Method {method_name} failed on {ds_name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    run_comparison()