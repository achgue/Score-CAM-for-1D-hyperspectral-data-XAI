import json
from src.dataset import UniversalSpectralDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

def load_data_matrix(dataset_path, class_map,  max_samples=1800, trim_start=20, trim_end=20):
    print(f"Caricamento dati da {dataset_path}...")
    
    # 1. Passiamo i parametri di trim DIRETTAMENTE al Dataset
    dataset = UniversalSpectralDataset(
        root_dir=dataset_path, 
        split='train', 
        class_map=class_map, 
        sample_number_per_class=max_samples,
        bands_path=None,
        trim_start=trim_start,  # <--- Sfrutta la logica interna
        trim_end=trim_end       # <--- Sfrutta la logica interna
    )
    
    # 2. Estrazione dati (sono GIA' trimmati)
    X_tensor = dataset.data
    y_tensor = dataset.labels

    # 3. Prendi pixel centrale (1,1) -> [N, Channels]
    X_center = X_tensor[:, :, 1, 1]

    # 4. Converti in Numpy
    X_np = X_center.numpy()
    y_np = y_tensor.numpy()
    
    print(f"   Converted to Numpy: X shape {X_np.shape}, y shape {y_np.shape}")

    # Non serve fare slicing manuale qui, X_np è già pulito!
    return X_np, y_np

def compute_snv(tensor_batch):
    """Applica SNV su un batch: (Batch, Channels) o (Batch, Channels, H, W)"""
    means = tensor_batch.mean(dim=1, keepdim=True)
    stds = tensor_batch.std(dim=1, keepdim=True)
    # Evita divisioni per zero aggiungendo un epsilon
    return (tensor_batch - means) / (stds + 1e-8)

def rescale_to_zero_one(tensor_batch):
    """Riscala ogni sample del batch tra 0 e 1."""
    mins = tensor_batch.min(dim=1, keepdim=True)[0]
    maxs = tensor_batch.max(dim=1, keepdim=True)[0]
    return (tensor_batch - mins) / (maxs - mins + 1e-8)

def generate_scatter_plots(INPUT_FILE, OUTPUT_DIR):
    # 1. Verifica esistenza file dati
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: Il file '{INPUT_FILE}' non esiste.")
        print("Esegui prima lo script di analisi globale per generarlo.")
        return

    # 2. Creazione cartella output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Cartella creata: {OUTPUT_DIR}")

    print("Caricamento dati...")
    # allow_pickle=True è necessario perché abbiamo salvato un dizionario
    data = np.load(INPUT_FILE, allow_pickle=True).item()
    
    print(f"Trovate {len(data)} classi. Inizio generazione grafici...")

    # 3. Loop su ogni classe
    for cls_name, cls_data in data.items():
        
        # Estrazione dati
        avg_spectrum = cls_data['avg_spectrum'] # Asse Y (Intensità)
        avg_saliency = cls_data['avg_saliency'] # Colore (Confidenza/Importanza)
        
        # Creiamo l'asse X (Bande/Frequenze)
        # Se hai una lista reale di lunghezze d'onda (es. nm), usala qui al posto di arange
        lWaveLength = np.arange(len(avg_spectrum)) 
        
        # Titolo del grafico
        strTitle = f"Saliency Scatter Plot - {cls_name}"

        # --- IL TUO CODICE DI PLOT (Adattato) ---
        plt.figure(figsize=(10, 6))
        
        # Scatter plot: X=Bande, Y=Spettro, Color=Saliency
        scatter = plt.scatter(lWaveLength, avg_spectrum, c=avg_saliency, cmap='jet', s=15)

        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Importance (Score-CAM)')

        # Add labels and title
        plt.xlabel('Spectral Bands (Index)')
        plt.ylabel('Reflectance Intensity (Avg)')
        plt.title(strTitle)
        
        plt.grid(True, alpha=0.3)

        # --- SALVATAGGIO ---
        # Sostituiamo caratteri scomodi nel nome file
        safe_name = cls_name.replace(" ", "_").replace("/", "-")
        save_path = os.path.join(OUTPUT_DIR, f"scatter_{safe_name}.png")
        
        plt.savefig(save_path, dpi=300) # dpi=300 per alta qualità tesi
        plt.close() # Chiude la figura per liberare memoria RAM
        
        print(f"Salvato: {save_path}")

    print("-" * 30)
    print("Finito! Tutti i grafici sono in 'results/scatter_plots'")

def select_bands_peaks_sorted(INPUT_FILE, OUTPUT_FILE_UNION, OUTPUT_FILE_INTERSECT, OUTPUT_FILE_PER_CLASS, OUTPUT_JSON, IGNORE_START=5, IGNORE_END=255, MIN_PROMINENCE=0.05, MIN_HEIGHT_REL=0.15, MIN_DISTANCE=5):

    if not os.path.exists(INPUT_FILE):
        print(f"Errore: File '{INPUT_FILE}' non trovato.")
        return

    data = np.load(INPUT_FILE, allow_pickle=True).item()
    
    final_bands_union = set()
    final_bands_intersection = None
    per_class_collection = {}
    
    print(f"\n--- SELEZIONE BASATA SU PICCHI (ORDINATI PER IMPORTANZA) ---")
    
    sorted_classes = sorted(data.items(), key=lambda item: item[1]['id'])
    
    for cls_name, cls_data in sorted_classes:
        original_saliency = cls_data['avg_saliency']
        saliency_proc = original_saliency.copy() 
        
        if saliency_proc.max() > 0:
            saliency_norm = saliency_proc / saliency_proc.max()
        else:
            continue
            
        print(f"\nClasse {cls_data['id']} ({cls_name}): Saliency normalizzata. Max: {saliency_norm.max():.4f}")
        
        # --- FIX 2: Adatta la distanza se il dataset è piccolo (es. 10 bande) ---
        current_distance = MIN_DISTANCE if len(saliency_norm) > 20 else 1
        
        # --- FIX 1: Padding per non perdere i picchi agli estremi (indice 0 o N-1) ---
        saliency_padded = np.pad(saliency_norm, (1, 1), mode='constant', constant_values=0)
        
        peaks_padded, _ = find_peaks(
            saliency_padded, 
            height=MIN_HEIGHT_REL, 
            distance=current_distance, 
            prominence=MIN_PROMINENCE
        )
        
        # Rimuoviamo l'offset causato dal padding per riavere gli indici corretti
        peaks = peaks_padded - 1 

        # Fallback nel caso in cui non trovi nulla
        if len(peaks) == 0:
            best_idx = np.argmax(saliency_proc)
            if saliency_proc[best_idx] > 0:
                peaks = np.array([best_idx])
            else:
                peaks = np.array([])

        # Costruzione Dati Dettagliati
        detailed_peaks = []
        for band_idx in peaks:
            band_idx = int(band_idx)
            score_val = float(original_saliency[band_idx])
            detailed_peaks.append({
                "band": band_idx,
                "score": score_val
            })
            
        # Ordinamento Decrescente per Score
        detailed_peaks.sort(key=lambda x: x['score'], reverse=True)

        # Salvataggio nel dizionario
        dict_key = f"{cls_data['id']} - {cls_name}"
        per_class_collection[dict_key] = detailed_peaks

        current_peaks_set = set([p['band'] for p in detailed_peaks])
        final_bands_union.update(current_peaks_set)

        if final_bands_intersection is None:
            final_bands_intersection = current_peaks_set
        else:
            final_bands_intersection = final_bands_intersection.intersection(current_peaks_set)

        top_band = detailed_peaks[0]['band'] if detailed_peaks else "N/A"
        print(f"  Trovati {len(detailed_peaks)} picchi. Top Band: {top_band}")

    if final_bands_intersection is None:
        final_bands_intersection = set()

    list_union = sorted(list(final_bands_union))
    list_intersect = sorted(list(final_bands_intersection))

    # Salvataggi
    with open(OUTPUT_FILE_UNION, 'w') as f:
        for band in list_union: f.write(f"{band}\n")
            
    with open(OUTPUT_FILE_INTERSECT, 'w') as f:
        for band in list_intersect: f.write(f"{band}\n")

    try:
        with open(OUTPUT_FILE_PER_CLASS, 'w') as f:
            f.write(f"--- CLASSFICA DELLE BANDE PIU' IMPORTANTI PER MATERIALE ---\n")
            f.write(f"Ordinamento: Score Decrescente (Dal più importante al meno)\n\n")
            
            for key, peak_data in per_class_collection.items():
                f.write(f"MATERIALE: {key}\n")
                f.write(f"{'Rank':<5} | {'Banda':<10} | {'Score (Importanza)':<20}\n")
                f.write("-" * 45 + "\n")
                for rank, item in enumerate(peak_data, 1):
                    f.write(f"{rank:<5} | {item['band']:<10} | {item['score']:.4f}\n")
                f.write("\n" + "="*45 + "\n\n")
        print(f"\n[OK] File classificato salvato in '{OUTPUT_FILE_PER_CLASS}'")
    except IOError as e: print(f"Errore Per-Class: {e}")

    # --- FIX 3: Modalità 'w' invece di 'x' ---
    try:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(per_class_collection, f, indent=4)
        print(f"[OK] JSON salvato in '{OUTPUT_JSON}'")
    except IOError as e: print(f"Errore JSON: {e}")