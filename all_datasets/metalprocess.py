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
    try:
        if path.lower().endswith('.hdr'):
            expected_bin_path = path[:-4] + '.bin'
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

def extract_and_save_class(img_padded, coords, pad, save_folder, prefix, chunk_size=2000):
    """Funzione helper per estrarre e salvare i chunk di una specifica classe."""
    total_pixels = len(coords)
    chunk_idx = 0
    
    for i in range(0, total_pixels, chunk_size):
        end_idx = min(i + chunk_size, total_pixels)
        batch_coords = coords[i:end_idx]
        current_batch_size = len(batch_coords)
        
        batch_patches = []
        for (r, c) in batch_coords:
            r_pad = r + pad
            c_pad = c + pad
            patch = img_padded[r_pad-pad : r_pad+pad+1, c_pad-pad : c_pad+pad+1, :]
            batch_patches.append(patch)
            
        batch_np = np.array(batch_patches, dtype=np.float32)
        batch_np = np.transpose(batch_np, (0, 3, 1, 2))  # (Batch, C, H, W)
        tensor_batch = torch.from_numpy(batch_np)
        
        # Nome file es: Metal_Immagine1_part0.pt
        filename = f"{prefix}_part{chunk_idx}.pt"
        save_path = os.path.join(save_folder, filename)
        
        torch.save(tensor_batch, save_path)
        print(f"      [{prefix}] Salvato {filename} (Samples: {current_batch_size})")
        
        chunk_idx += 1
        del batch_patches, batch_np, tensor_batch
        gc.collect()
        
    return total_pixels

def process_and_save_chunks(img, mask, patch_size, save_folder, base_name, total_target_samples):
    h, w, c = img.shape
    pad = patch_size // 2
    
    rows_metal, cols_metal = np.where(mask > 0)
    rows_bg, cols_bg = np.where(mask == 0)
    
    coords_metal = list(zip(rows_metal, cols_metal))
    coords_bg = list(zip(rows_bg, cols_bg))
    
    np.random.shuffle(coords_metal)
    np.random.shuffle(coords_bg)
    
    target_per_class = total_target_samples // 2
    limit = min(target_per_class, len(coords_metal), len(coords_bg))
    
    if limit == 0:
        print("   -> ATTENZIONE: Nessun pixel trovato per una delle due classi!")
        return 0, 0
    
    print(f"   -> Trovati {len(coords_metal)} metallo, {len(coords_bg)} sfondo.")
    print(f"   -> Estrazione di {limit} sample per classe (Totale: {limit*2})...")
    
    selected_metal = coords_metal[:limit]
    selected_bg = coords_bg[:limit]
    
    # Padding immagine
    img_padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    # Estraiamo e salviamo le due classi separatamente
    extract_and_save_class(img_padded, selected_metal, pad, save_folder, prefix=f"Metal_{base_name}")
    extract_and_save_class(img_padded, selected_bg, pad, save_folder, prefix=f"Background_{base_name}")
    
    return limit * 2, 2

def process_hsi_envi(input_dir, output_dir, mask_dir, patch_size=1, total_samples=10000):
    if not os.path.exists(mask_dir):
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Cerca file TIFF o HDR nella cartella input
    hsi_files = glob.glob(os.path.join(input_dir, "*.hdr")) + glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))

    for hsi_path in hsi_files:
        filename_stem = Path(hsi_path).stem 
        
        # --- MODIFICA CHIAVE: Cerca la maschera con lo stesso identico nome ma .png ---
        found_mask_path = os.path.join(mask_dir, f"{filename_stem}.png")
        
        if not os.path.exists(found_mask_path):
            print(f"[SKIP] Maschera non trovata per l'immagine: {filename_stem}")
            continue

        try:
            print(f"\n--- Processing: {filename_stem} ---")
            img_data = load_robust_image(hsi_path)
            mask_data = load_robust_image(found_mask_path)

            if img_data.shape[0] != mask_data.shape[0]: 
                img_data = np.transpose(img_data, (1, 2, 0))

            h, w = img_data.shape[:2]
            if img_data.ndim == 3 and img_data.shape[0] < h and img_data.shape[0] < w:
                 img_data = np.transpose(img_data, (1, 2, 0))
                 h, w = img_data.shape[:2]
            
            # Assicuriamoci che la maschera sia 2D
            if mask_data.ndim == 3: mask_data = mask_data[:, :, 0]
            
            save_folder = os.path.join(output_dir, filename_stem)
            os.makedirs(save_folder, exist_ok=True)
            
            process_and_save_chunks(img_data, mask_data, patch_size, save_folder, filename_stem, total_samples)
            gc.collect()

        except Exception as e:
            print(f"[ERRORE] Fallito su {filename_stem}: {e}")