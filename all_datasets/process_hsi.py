import os
import glob
import torch
import tifffile
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from spectral.io import envi
import gc

def load_robust_image(path):
    """
    Carica l'immagine gestendo formati ENVI (.hdr) e TIFF.
    """
    try:
        if path.lower().endswith('.hdr'):
            expected_bin_path = path[:-4] + '.bin'
            # Cerca il binario con o senza estensione
            if not os.path.exists(expected_bin_path):
                path_no_ext = path[:-4]
                if os.path.exists(path_no_ext):
                    expected_bin_path = path_no_ext

            if os.path.exists(expected_bin_path):
                img_obj = envi.open(path, expected_bin_path)
            else:
                img_obj = envi.open(path)
            
            return np.array(img_obj.load())

        elif path.lower().endswith(('.tif', '.tiff')):
            return tifffile.imread(path)
        else:
            return iio.imread(path)

    except Exception as e:
        raise e

def process_and_save_chunks(img, mask, patch_size, save_folder, base_name, chunk_size=3000):
    """
    Estrae le patch e salva IMMEDIATAMENTE ogni chunk su disco.
    Formato file: Nome_Patch_PartIndex_NumSamples.pt
    """
    h, w, c = img.shape
    pad = patch_size // 2
    
    # 1. Identifica i pixel validi
    rows, cols = np.where(mask > 0)
    total_pixels = len(rows)
    print(f"   -> Trovati {total_pixels} pixel validi. Inizio salvataggio a blocchi...")

    # 2. Padding dell'immagine (unica operazione pesante in memoria)
    # Usiamo mode='edge' per replicare i bordi ed evitare artefatti neri
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    # Iteratore per l'indice del chunk (part_0, part_1, etc.)
    chunk_idx = 0
    
    # 3. Ciclo sui blocchi
    for i in range(0, total_pixels, chunk_size):
        
        # Definisci inizio e fine del blocco corrente
        end_idx = min(i + chunk_size, total_pixels)
        batch_rows = rows[i:end_idx]
        batch_cols = cols[i:end_idx]
        current_batch_size = len(batch_rows)
        
        batch_patches = []
        
        # Estrazione patch per il blocco corrente
        for r, c in zip(batch_rows, batch_cols):
            # Shift coordinate dovuto al padding
            r_pad = r + pad
            c_pad = c + pad
            
            # Slice numpy
            patch = img_padded[r_pad-pad : r_pad+pad+1, c_pad-pad : c_pad+pad+1, :]
            batch_patches.append(patch)
        
        # Conversione in Tensore
        # Numpy array: (Batch, H, W, C)
        batch_np = np.array(batch_patches, dtype=np.float32)
        
        # Transpose per PyTorch: (Batch, C, H, W)
        batch_np = np.transpose(batch_np, (0, 3, 1, 2))
        tensor_batch = torch.from_numpy(batch_np)
        
        # --- SALVATAGGIO IMMEDIATO ---
        # Nome file: NomeClasse_DimPatch_PartX_NumSamples.pt
        # Esempio: Roofs_p5_part0_2000.pt
        filename = f"{base_name}_p{patch_size}_part{chunk_idx}_{current_batch_size}.pt"
        save_path = os.path.join(save_folder, filename)
        
        torch.save(tensor_batch, save_path)
        
        # Feedback console
        print(f"      [Part {chunk_idx}] Salvato {filename}")
        
        # --- PULIZIA MEMORIA ---
        chunk_idx += 1
        del batch_patches, batch_np, tensor_batch
        gc.collect() # Forza il garbage collector
        
    return total_pixels, chunk_idx

def process_hsi_envi(input_dir, output_dir, mask_dir, patch_size=1):
    
    if not os.path.exists(mask_dir):
        print(f"Errore: Cartella maschere '{mask_dir}' non trovata.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    hsi_files = glob.glob(os.path.join(input_dir, "*.hdr")) + \
                glob.glob(os.path.join(input_dir, "*.tif"))
    
    if not hsi_files:
        print("Nessun file trovato.")
        return

    print(f"Inizio elaborazione di {len(hsi_files)} file. Patch Size: {patch_size}")

    mask_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for hsi_path in hsi_files:
        filename_stem = Path(hsi_path).stem 
        
        # Cerca maschera
        found_mask_path = None
        for ext in mask_extensions:
            mask_name = f"{filename_stem}_label{ext}"
            potential_mask = os.path.join(mask_dir, mask_name)
            if os.path.exists(potential_mask):
                found_mask_path = potential_mask
                break
        
        if found_mask_path is None:
            continue

        try:
            print(f"--- Processing: {filename_stem} ---")
            
            # Carica dati
            img_data = load_robust_image(hsi_path)
            mask_data = load_robust_image(found_mask_path)

            # fix formato (C, H, W) -> (H, W, C) se necessario
            if img_data.shape[0] != mask_data.shape[0]: 
                print("   -> Rilevato formato (C, H, W). Trasposizione in (H, W, C)...")
                # Da (C, H, W) (0, 1, 2) a (H, W, C) (1, 2, 0)
                img_data = np.transpose(img_data, (1, 2, 0))

            print(f"Caricati: Immagine {img_data.shape}, Maschera {mask_data.shape}")
            
            # Gestione dimensioni
            h, w = img_data.shape[:2]
            # Fix canali prima (C, H, W) -> (H, W, C)
            if img_data.ndim == 3 and img_data.shape[0] < h and img_data.shape[0] < w:
                 img_data = np.transpose(img_data, (1, 2, 0))
                 h, w = img_data.shape[:2]
            
            bands = img_data.shape[2] if img_data.ndim == 3 else 1
            if mask_data.ndim == 3: mask_data = mask_data[:, :, 0]
            
            # Cartella specifica per questo file HSI
            save_folder = os.path.join(output_dir, filename_stem)
            os.makedirs(save_folder, exist_ok=True)
            
            # --- CHIAMATA ALLA FUNZIONE CHUNKED ---
            # chunk_size=2000 è conservativo per evitare OOM. Puoi alzarlo a 5000 se hai 16GB+ RAM.
            total_samples, num_parts = process_and_save_chunks(
                img_data, 
                mask_data, 
                patch_size, 
                save_folder, 
                base_name=filename_stem,
                chunk_size=2000 
            )
            
            # Salva metadati globali
            meta = {
                'source_file': filename_stem,
                'orig_shape': (h, w, bands),
                'patch_size': patch_size,
                'total_samples': total_samples,
                'num_parts': num_parts,
                'data_format': 'split_files'
            }
            torch.save(meta, os.path.join(save_folder, "meta.pt"))
            
            # Pulizia finale prima del prossimo file
            del img_data, mask_data
            gc.collect()

        except Exception as e:
            print(f"[ERRORE] Fallito su {filename_stem}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    BASE_DIR = "./dataset_baupol"
    INPUT_FOLDER = os.path.join(BASE_DIR, "raw_data")
    MASK_FOLDER = os.path.join(BASE_DIR, "masks")
    OUTPUT_FOLDER = "./dataset_baupol_patches"
    
    # Imposta la dimensione della patch
    PATCH_SIZE = 3 
    
    process_hsi_envi(INPUT_FOLDER, OUTPUT_FOLDER, MASK_FOLDER, patch_size=PATCH_SIZE)