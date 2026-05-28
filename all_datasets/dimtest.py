import torch

import glob

import os



# --- CONFIGURAZIONE ---

# Assicurati che questo percorso punti alla tua cartella di output

SEARCH_DIR = "output_pt/baumpoly/model_train_split/test" 

# Se vuoi filtrare solo la classe che dà problemi, scrivi parte del nome qui sotto (es. "Papier_Leinen")

FILTER_NAME = "TShirt_Baumwolle_1Norm_test.pt" 

# ----------------------



print(f"--- Controllo integrità dataset in: {SEARCH_DIR} ---")



# Cerca tutti i file .pt ricorsivamente

files = glob.glob(os.path.join(SEARCH_DIR, "**", "*.pt"), recursive=True)

files.sort() # Ordina per avere una visualizzazione pulita



if not files:

    print("❌ Nessun file .pt trovato. Controlla il percorso SEARCH_DIR.")



count = 0

for f_path in files:

    fname = os.path.basename(f_path)

    

    # Filtro opzionale per debuggare solo file specifici

    if FILTER_NAME and FILTER_NAME not in fname:

        continue



    try:

        # Carica il file

        data = torch.load(f_path)

        

        # Caso 1: È il file dei metadati (dizionario)

        if isinstance(data, dict):

            print(f"\n📄 [METADATA] {fname}")

            if 'orig_shape' in data:

                print(f"   Orig Image Shape: {data['orig_shape']}")

                print(f"   Total Samples:    {data.get('total_samples', 'N/A')}")



        # Caso 2: È un chunk di dati (Tensore PyTorch)

        elif isinstance(data, torch.Tensor):

            print(f"\n📦 [DATA CHUNK] {fname}")

            print(f"   Dataset Shape (Batch):   {data.shape}")       # Es: (2000, 100, 3, 3)

            print(f"   Element Shape (Single):  {data[0].shape}")    # Es: (100, 3, 3)
            print(f"   Content:  {data[0]}")    # Es: (100, 3, 3)

            print(f"   Tipo Dati:               {data.dtype}")

            

            # Controllo di sanità extra

            if data.ndim != 4:

                print(f"   ⚠️ WARNING: Atteso 4 dimensioni (N, C, H, W), trovate {data.ndim}")



        count += 1

        # Rimuovi il break se vuoi vedere TUTTI i file, altrimenti si ferma ai primi 5

        if count >= 5 and not FILTER_NAME:

            print("\n... Interrotto dopo 5 file (rimuovi il break nello script per vederli tutti) ...")

            break



    except Exception as e:

        print(f"\n❌ ERRORE leggendo {fname}: {e}")