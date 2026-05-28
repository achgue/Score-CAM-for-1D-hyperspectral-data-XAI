import numpy as np
import torch
import tifffile
import imageio
import matplotlib.colors as mcolors
import cv2
from tqdm import tqdm
import os
from src.models import MLP

# 1. IMPORTA LA TUA CONFIGURAZIONE
import configs.datasets as config_datasets

# --- CONFIGURAZIONE PATH GLOBALI ---
DATASET_NAME = 'tanks_uniud' 
MODEL_WEIGHTS_PATH = 'checkpoints/tanks_uniud/3_bands_MLP_best_98.25.pt' 

# LISTA DEGLI ID DA PROCESSARE
# Aggiungi qui tutti gli ID che vuoi classificare
ID_LIST = [
    "IMG_0111",
    "IMG_0524",
    "IMG_0558",
    "IMG_0555",
    "IMG_0821",
]

# ==============================================================================

def prepare_colormap(dataset_config):
    """Prepara l'array dei colori RGB con lo sfondo nero all'ultimo indice."""
    num_classes = len(dataset_config['class_map'])
    colormap = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    colormap[num_classes] = [0, 0, 0] # Sfondo Nero
    
    if 'class_colors' in dataset_config:
        for class_id, hex_code in dataset_config['class_colors'].items():
            colormap[class_id] = tuple(int(c * 255) for c in mcolors.to_rgb(hex_code))
    else:
        colormap[:num_classes] = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

    return colormap, num_classes

def load_selected_bands(filepath):
    """Legge e ordina gli indici delle bande."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File delle bande non trovato: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read().replace(',', ' ').split()
        
    bands = [int(b) for b in content]
    bands = sorted(list(set(bands))) # FONDAMENTALE PER L'ALLINEAMENTO
    return bands

def process_single_image(img_id, model, device, colormap, num_classes, selected_bands):
    """
    Funzione dedicata all'elaborazione di una singola immagine.
    """
    print(f"\n{'='*50}")
    print(f"INIZIO ELABORAZIONE: {img_id}")
    print(f"{'='*50}")
    
    input_tiff_path = f'../all_datasets/raw/tanks_uniud/images/{img_id}.tiff' 
    input_mask_path = f'../all_datasets/raw/tanks_uniud/masks/{img_id}.png' 
    output_jpeg_path = f'classification_map_{img_id}.jpg'
    
    # 1. Controlli di esistenza file
    if not os.path.exists(input_tiff_path):
        print(f"[ERRORE] TIFF non trovato: {input_tiff_path}. Salto...")
        return
    if not os.path.exists(input_mask_path):
        print(f"[ERRORE] Maschera non trovata: {input_mask_path}. Salto...")
        return

    # 2. Caricamento TIFF
    print("Caricamento TIFF...")
    image_cube = tifffile.imread(input_tiff_path) 
    if image_cube.shape[0] < image_cube.shape[2]: 
        image_cube = np.transpose(image_cube, (1, 2, 0))
    H, W, _ = image_cube.shape

    # 3. Riduzione Dimensionale
    image_cube = image_cube[:, :, selected_bands]
    
    # 4. Caricamento Maschera
    mask_img = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img.shape[:2] != (H, W):
        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_img > 127

    # 5. Padding
    padded_image = np.pad(image_cube, pad_width=((1, 1), (1, 1), (0, 0)), mode='reflect')
    prediction_map = np.full((H, W), num_classes, dtype=np.int32)

    # 6. Classificazione
    print("Inizio scansione spaziale...")
    with torch.no_grad():
        for i in tqdm(range(H), desc=f"Righe {img_id}"):
            row_patches = []
            valid_cols = []
            
            for j in range(W):
                if not mask_bool[i, j]:
                    continue 
                
                patch = padded_image[i:i+3, j:j+3, :]
                row_patches.append(patch)
                valid_cols.append(j)
            
            if len(row_patches) == 0:
                continue
                
            batch_tensor = torch.tensor(np.array(row_patches), dtype=torch.float32)
            batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(device)
            batch_tensor = batch_tensor.contiguous()

            # Inferenza Reale
            outputs = model(batch_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            prediction_map[i, valid_cols] = preds

    # 7. Colorazione e Salvataggio
    colored_image = colormap[prediction_map]
    print(f"Salvataggio in: {output_jpeg_path}")
    imageio.imwrite(output_jpeg_path, colored_image, quality=95)
    print("Elaborazione completata con successo.")


def main():
    # A. SETUP INIZIALE E COLORI
    dataset_config = next(item for item in config_datasets.DATASET_CATALOG if item['name'] == DATASET_NAME)
    colormap, num_classes = prepare_colormap(dataset_config)
    device = torch.device(config_datasets.TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu')

    # B. CARICAMENTO BANDE
    bands_file_path = dataset_config.get('bands_file')
    selected_bands = load_selected_bands(bands_file_path)
    reduced_bands_count = len(selected_bands)
    print(f"Caricate {reduced_bands_count} bande. Indici: {selected_bands}")

    # C. CARICAMENTO MODELLO GLOBALE (Una volta sola per tutti gli ID)
    print(f"\nCaricamento modello su {device}...")
    model = MLP(input_channels=reduced_bands_count, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()
    print("Modello caricato e pronto per il batch processing.")

    # D. CICLO SULLA LISTA DEGLI ID
    for img_id in ID_LIST:
        process_single_image(
            img_id=img_id, 
            model=model, 
            device=device, 
            colormap=colormap, 
            num_classes=num_classes, 
            selected_bands=selected_bands
        )
        
    print("\n[SUCCESSO] Tutte le immagini nella lista sono state processate!")

if __name__ == '__main__':
    main()