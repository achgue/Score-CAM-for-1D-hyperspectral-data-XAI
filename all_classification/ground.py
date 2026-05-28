import numpy as np
import imageio
import matplotlib.colors as mcolors
import cv2
import os

# 1. IMPORTA LA TUA CONFIGURAZIONE
import configs.datasets as config_datasets

# --- CONFIGURAZIONE PATH GLOBALI ---
DATASET_NAME = 'baumlein' 

# LISTA DEGLI ID DA PROCESSARE (La stessa usata per l'inferenza)
ID_LIST = [
    "Papier_Baumwolle_1_000000Norm",
    "Papier_Baumwolle_2_000000Norm",
    "Papier_Baumwolle_3_000000Norm",
    "Papier_Baumwolle_4_000000Norm",
    "Papier_Baumwolle_5_000000Norm",
    "Papier_Baumwolle_6_000000Norm",
    "Papier_Baumwolle_7_000000Norm",
    "Papier_Baumwolle_8_000000Norm",
    "Papier_Leinen_222_000000Norm",
    "Papier_Leinen_269_000000Norm",
    "Papier_Leinen_275OE_grau_000000Norm",
    "Papier_Leinen_275OE_Weiss_000000Norm",
    "Papier_Leinen_316_000000Norm",
    "Papier_Leinen_504RL_000000Norm",
    "Papier_Leinen_712Z_000000Norm",
    "Papier_Leinen_816_Weiss_000000Norm"
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

def get_class_id_from_filename(filename, class_map):
    """
    Cerca il nome della classe all'interno del nome del file
    restituendo l'indice (ID) numerico corrispondente.
    """
    for class_name, class_idx in class_map.items():
        if class_name in filename:
            return class_idx
    return None

def process_single_gt(img_id, colormap, num_classes, class_map):
    """
    Genera il Ground Truth per una singola immagine.
    """
    print(f"Elaborazione GT: {img_id}")
    
    input_mask_path = f'../all_datasets/raw/baumlein/masks/{img_id}_label.png' 
    output_jpeg_path = f'ground_truth_map_{img_id}.jpg'
    
    # 1. Controlli
    if not os.path.exists(input_mask_path):
        print(f"  [ERRORE] Maschera non trovata: {input_mask_path}. Salto...")
        return

    class_idx = get_class_id_from_filename(img_id, class_map)
    if class_idx is None:
        print(f"  [ERRORE] Impossibile dedurre la classe per {img_id}. Salto...")
        return

    # 2. Caricamento Maschera
    mask_img = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"  [ERRORE] Impossibile leggere l'immagine {input_mask_path}.")
        return
        
    H, W = mask_img.shape[:2]
    mask_bool = mask_img > 127

    # 3. Creazione della mappa Ground Truth
    # Inizializza tutto allo sfondo (ultimo indice)
    gt_map = np.full((H, W), num_classes, dtype=np.int32)
    
    # Sostituisce lo sfondo con l'indice della classe, ma SOLO dove la maschera è bianca
    gt_map[mask_bool] = class_idx

    # 4. Colorazione e Salvataggio
    colored_image = colormap[gt_map]
    imageio.imwrite(output_jpeg_path, colored_image, quality=95)
    print(f"  -> Salvato come {output_jpeg_path} (Classe: {class_idx})")


def main():
    # A. SETUP INIZIALE E COLORI
    dataset_config = next(item for item in config_datasets.DATASET_CATALOG if item['name'] == DATASET_NAME)
    colormap, num_classes = prepare_colormap(dataset_config)
    class_map = dataset_config['class_map']

    print("="*50)
    print(" GENERAZIONE GROUND TRUTH MAPS")
    print("="*50)

    # B. CICLO SULLA LISTA DEGLI ID
    for img_id in ID_LIST:
        process_single_gt(
            img_id=img_id, 
            colormap=colormap, 
            num_classes=num_classes, 
            class_map=class_map
        )
        
    print("\n[SUCCESSO] Tutte le immagini Ground Truth sono state generate!")

if __name__ == '__main__':
    main()