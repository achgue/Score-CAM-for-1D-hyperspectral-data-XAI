import os
import sys

# Importiamo le funzioni dai file che abbiamo creato in precedenza
# Assicurati che process_hsi.py e split_dataset.py siano nella stessa cartella di questo script
try:
    from metalprocess import process_hsi_envi
    from metalsplit import split_dataset
except ImportError as e:
    print("ERRORE CRITICO: Non trovo i file 'process_hsi.py' o 'split_dataset.py'.")
    print(f"Dettaglio: {e}")
    sys.exit(1)

def run_pipeline(target_folders, base_raw_dir, base_output_dir, patch_size, split_ratios):
    """
    Esegue la pipeline completa (Generazione Patch -> Split Dataset) per una lista di cartelle.
    """
    
    print(f"=== AVVIO PIPELINE ===")
    print(f"Cartelle target: {target_folders}")
    print(f"Patch Size: {patch_size}")
    print(f"Split Ratios: {split_ratios}\n")

    for folder_name in target_folders:
        print(f"--------------------------------------------------")
        print(f">>> PROCESSING CATEGORIA: {folder_name.upper()}")
        print(f"--------------------------------------------------")

        # 1. COSTRUZIONE DEI PERCORSI (basata sulla tua struttura)
        # Input: raw/{folder}/images e raw/{folder}/masks
        path_raw_images = os.path.join(base_raw_dir, folder_name, "images")
        path_raw_masks = os.path.join(base_raw_dir, folder_name, "masks")

        # Output Intermedio: output_pt/{folder}/patches_split
        path_patches_out = os.path.join(base_output_dir, folder_name, "patches_split")

        # Output Finale: output_pt/{folder}/model_train_split
        path_split_out = os.path.join(base_output_dir, folder_name, "model_train_split")

        # 2. CONTROLLI DI ESISTENZA
        if not os.path.exists(path_raw_images):
            print(f"[SKIP] Cartella immagini non trovata: {path_raw_images}")
            continue
        
        if not os.path.exists(path_raw_masks):
            print(f"[SKIP] Cartella maschere non trovata: {path_raw_masks}")
            continue

        # 3. STEP 1: GENERAZIONE PATCHES (process_hsi)
        print(f"\n[FASE 1] Generazione Patch ({patch_size}x{patch_size})...")
        try:
            process_hsi_envi(
                input_dir=path_raw_images,
                output_dir=path_patches_out,
                mask_dir=path_raw_masks,
                patch_size=patch_size
            )
        except Exception as e:
            print(f"[ERRORE FASE 1] Fallito process_hsi per {folder_name}: {e}")
            continue # Se fallisce la patch generation, inutile fare lo split

        # 4. STEP 2: DATASET SPLIT (split_dataset)
        print(f"\n[FASE 2] Splitting Dataset (Train/Val/Test)...")
        try:
            # Controlliamo se la fase 1 ha prodotto qualcosa
            if not os.path.exists(path_patches_out) or not os.listdir(path_patches_out):
                print(f"[WARN] Nessuna patch trovata in {path_patches_out}. Salto lo split.")
                continue

            split_dataset(
                input_dir=path_patches_out,
                output_dir=path_split_out,
                split_ratios=split_ratios
            )
        except Exception as e:
            print(f"[ERRORE FASE 2] Fallito split_dataset per {folder_name}: {e}")

    print("\n=== PIPELINE COMPLETATA ===")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    
    # 1. Cartelle Base (Root)
    BASE_RAW = "./raw"
    BASE_OUTPUT = "./output_pt"

    # 2. Lista delle cartelle da processare (nomi delle sottocartelle in raw)
    # Puoi commentare quelle che non vuoi processare
    TARGET_FOLDERS_LIST = [
        #"baumpoly",
        #"baumlein", 
        #"plastics",
        "tanks_uniud"
    ]

    # 3. Parametri Iperspettrali
    PATCH_SIZE = 3  # Dimensione della patch (es. 5x5, 3x3)

    # 4. Parametri di Split
    # (Train, Validation, Test) - La somma deve fare 1.0
    RATIOS = (0.70, 0.15, 0.15)

    # --- ESECUZIONE ---
    run_pipeline(
        target_folders=TARGET_FOLDERS_LIST,
        base_raw_dir=BASE_RAW,
        base_output_dir=BASE_OUTPUT,
        patch_size=PATCH_SIZE,
        split_ratios=RATIOS
    )