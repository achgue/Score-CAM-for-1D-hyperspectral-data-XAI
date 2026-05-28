import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- Import dei tuoi moduli custom ---
from cam.scorecam1D import ScoreCAM1D
from src.dataset import UniversalSpectralDataset
from src.models import get_model # O from src.models.architecture.CNN_1D import SpectralCNN1D
from configuration import SCORECAM_PARAMS, PREPROCESSING_CONFIG, DATASET_CATALOG, MODELS
from src.utils import compute_snv, rescale_to_zero_one, generate_scatter_plots, select_bands_peaks_sorted


def run_scorecam_pipeline():
    device = torch.device(SCORECAM_PARAMS['device'] if torch.cuda.is_available() else "cpu")
    print(f"Inizializzazione Score-CAM Pipeline su: {device}")
    
    os.makedirs(SCORECAM_PARAMS['output_dir'], exist_ok=True)

    # --- CICLO SUI DATASET ---
    for ds_conf in DATASET_CATALOG:
        ds_name = ds_conf['name']
        class_map = ds_conf['class_map']
        num_classes = len(class_map)
        checkpoint_path = ds_conf['checkpoint_path']
        
        # Mappa inversa per avere i nomi veri (es. 0 -> "Baumwolle_1")
        idx_to_class_name = {v: k for k, v in class_map.items()}

        print(f"\n{'='*50}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*50}")

        # 1. Caricamento Dati (Test o Train split, dipende da cosa vuoi analizzare)
        try:
            dataset = UniversalSpectralDataset(
                root_dir=ds_conf['path'], 
                split='test', # <-- Usa il set di test per l'interpretazione!
                class_map=class_map,
                sample_number_per_class=SCORECAM_PARAMS['max_samples_per_class'],
                bands_path=None, # Tutte le bande disponibili dopo il trim
                #trim_start=PREPROCESSING_CONFIG['trim_start'],
                #trim_end=PREPROCESSING_CONFIG['trim_end']
            )
            
            in_channels = dataset.current_channels
            print(f"Dataset caricato: {len(dataset)} campioni, {in_channels} canali.")
            
            # DataLoader con batch_size=1 (Score-CAM di solito si calcola su 1 sample alla volta)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            
        except Exception as e:
            print(f"Skipping dataset {ds_name} error: {e}")
            continue

        # --- CICLO SUI MODELLI ---
        # Per ora analizziamo solo SpectralCNN1D, ma il ciclo ci permette di estenderlo
        for model_name, model_conf in MODELS.items():
            print(f"\n   --- Analisi Modello: {model_name} ---")
            
            if not os.path.exists(checkpoint_path):
                print(f"   [Errore] Checkpoint non trovato: {checkpoint_path}. Skipping.")
                continue

            # 2. Istanziazione e Caricamento Modello
            model = get_model(model_name, in_channels, num_classes).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()

            # 3. Setup Score-CAM
            target_layer = model_conf['target_layer_name']
            model_dict = dict(type='cnn1d', arch=model, layer_name=target_layer, input_size=(in_channels,))
            try:
                scorecam = ScoreCAM1D(model_dict)
            except Exception as e:
                 print(f"   [Errore] Fallita inizializzazione ScoreCAM per {model_name}: {e}")
                 continue

            # 4. Inizializzazione Accumulatori (Usiamo defaultdict per comodità)
            saliency_sums = defaultdict(lambda: np.zeros(in_channels))
            spectrum_sums = defaultdict(lambda: np.zeros(in_channels))
            class_counts = defaultdict(int)

            # 5. Iterazione sui dati
            print(f"   Esecuzione Score-CAM ({len(dataset)} iterazioni)...")
            
            for X_batch, y_batch in tqdm(dataloader, desc="Score-CAM"):

                # --- PREPROCESSING ---
                # Estraiamo il pixel centrale e facciamo unsqueeze per avere [1, 1, Canali]
                # Questo DEVE corrispondere esattamente a cosa fa il tuo forward() nel modello
                #X_prep = rescale_to_zero_one(compute_snv(X_batch))
                #X_input = X_prep.to(device).float()
                #X_prep.unsqueeze(1).to(device).float() # [1, 1, Channels]
                X_input = X_batch.to(device).float()
                
                # --- PREDIZIONE ---
                with torch.no_grad():
                    #print(X_input.shape)
                    output = model(X_input)
                    predicted_class = output.max(1)[-1].item()
                
                # --- SCORE-CAM ---
                # Generiamo la saliency map spiegando la classe PREVISTA dal modello
                saliency_map = scorecam(X_input, class_idx=predicted_class)
                
                if saliency_map is None: 
                    continue
                
                # Estrazione array 1D
                sal_np = saliency_map.squeeze().cpu().detach().numpy()
                spec_np = X_input.squeeze().cpu().detach().numpy() # Salviamo lo spettro preprocessato
                spec_np = spec_np[:,1,1] 
                
                # Accumulo
                saliency_sums[predicted_class] += sal_np
                spectrum_sums[predicted_class] += spec_np
                class_counts[predicted_class] += 1

            # 6. Aggregazione e Salvataggio Risultati
            print("   Aggregazione risultati...")
            results_data = {}
            
            for cls_idx, count in class_counts.items():
                if count == 0: continue
                
                cls_name = idx_to_class_name.get(cls_idx, f"Class_{cls_idx}")
                
                avg_saliency = saliency_sums[cls_idx] / count
                avg_spectrum = spectrum_sums[cls_idx] / count
                
                # Qui NON applichiamo l'offset (+ trim_start) perché questi vettori 
                # (saliency e spectrum) sono lunghi esattamente quanto il dataset trimmato.
                # Se vogliamo sapere a che banda originale corrisponde il picco della saliency, 
                # aggiungeremo l'offset nel momento in cui faremo il plot o calcoleremo l'argmax.

                results_data[cls_name] = {
                    'id': cls_idx,
                    'count': count,
                    'avg_saliency': avg_saliency,
                    'avg_spectrum': avg_spectrum
                }

            ## Nome file: results/scorecam/saliency_baumlein_SpectralCNN1D.npy
            save_name = f"saliency_scores.npy"
            saliency_path = os.path.join(SCORECAM_PARAMS['output_dir'], f"{model_name}", f"{ds_name}", save_name)
            os.makedirs(os.path.dirname(saliency_path), exist_ok=True)
            np.save(saliency_path, results_data)
            #
            print(f"   [OK] Dati salvati in: {saliency_path}")

            select_bands_path = os.path.join(SCORECAM_PARAMS['output_dir'], f"{model_name}", f"{ds_name}", 'selected_bands')
            os.makedirs(select_bands_path, exist_ok=True)

            generate_scatter_plots(saliency_path, os.path.join(SCORECAM_PARAMS['output_dir'], f"{model_name}", f"{ds_name}", 'scatter_plots'))
            select_bands_peaks_sorted(saliency_path, os.path.join(select_bands_path, "union.txt"), os.path.join(select_bands_path, "intersection.txt"), os.path.join(select_bands_path, "peaks_per_class.txt"), os.path.join(select_bands_path, "peaks_dict_ranked.json"))


if __name__ == "__main__":
    run_scorecam_pipeline()