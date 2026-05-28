import os
import torch
from torch.utils.data import Dataset
import glob

class UniversalSpectralDataset(Dataset):
    def __init__(self, root_dir, split='train', class_map=None, 
                 sample_number_per_class=1800, bands_path=None, 
                 trim_start=0, trim_end=0):
        """
        root_dir: path alla cartella dati
        class_map: dizionario {'nome_classe': int_label}
        sample_number_per_class: int, target esatto di campioni per classe
        bands_path: path al file .txt con gli indici delle bande da tenere
        trim_start: int, quante bande rimuovere all'inizio (rumore)
        trim_end: int, quante bande rimuovere alla fine (rumore)
        """
        self.class_map = class_map

        # --- LOGICA LETTURA FILE BANDE ---
        self.selected_bands = None
        if bands_path and os.path.exists(bands_path):
            print(f"   [Info] Reading bands from {bands_path}...")
            try:
                with open(bands_path, 'r') as f:
                    # Legge le linee, rimuove spazi vuoti, converte in int
                    indices = [int(line.strip()) for line in f if line.strip().isdigit()]
                
                if indices:
                    # Converte in LongTensor per usare come indice avanzato in PyTorch
                    self.selected_bands = torch.tensor(indices, dtype=torch.long)
                    print(f"   [Info] Selected {len(self.selected_bands)} bands.")
                else:
                    print("   [Warning] Bands file is empty. Using ALL bands.")
            except Exception as e:
                print(f"   [Error] Could not read bands file: {e}. Using ALL bands.")
        
        # --- CARICAMENTO LISTA FILE ---
        search_path = os.path.join(root_dir, split, '*.pt')
        all_files = glob.glob(search_path)
        
        if not all_files:
            raise FileNotFoundError(f"Nessun file trovato in {search_path}")

        print(f"--- Caricamento Dataset: {split} ---")
        print(f"Target: {sample_number_per_class} campioni per classe.")
        if trim_start > 0 or trim_end > 0:
            print(f"Trimming attivo: Start={trim_start}, End={trim_end}")

        # 1. Organizza i file per classe
        files_by_class = {v: [] for k, v in class_map.items()}
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            found = False
            for class_name, class_idx in class_map.items():
                if class_name in filename:
                    files_by_class[class_idx].append(file_path)
                    found = True
                    break
            if not found:
                print(f"[Warning] Ignorato file non mappato: {filename}")

        # Liste per accumulare i dati finali
        final_data_list = []
        final_labels_list = []

        # 2. Itera su ogni classe
        for class_idx, file_list in files_by_class.items():
            if not file_list:
                print(f"   Classe {class_idx}: Nessun file trovato.")
                continue

            # Accumulatore temporaneo per QUESTA classe
            class_tensors = []
            current_count = 0
            
            for file_path in file_list:
                # Se abbiamo già raggiunto il target per questa classe, stop
                if sample_number_per_class is not None and current_count >= sample_number_per_class:
                    break
                
                try:
                    # Carica il blocco: [N_samples, Channels, H, W]
                    chunk = torch.load(file_path)
                    
                    # === 1. APPLICA IL TRIMMING (RIMOZIONE RUMORE) ===
                    # Questo va fatto PRIMA della selezione bande, per allineare gli indici
                    if trim_start > 0 or trim_end > 0:
                        n_channels_raw = chunk.shape[1]
                        end_idx = n_channels_raw - trim_end
                        
                        # Controllo di sicurezza
                        if end_idx <= trim_start:
                             raise ValueError(f"Errore Trimming: trim_start ({trim_start}) >= canali totali - trim_end ({end_idx})")

                        # Taglio effettivo (slice)
                        chunk = chunk[:, trim_start:end_idx, :, :]

                    # === 2. APPLICA LA SELEZIONE BANDE ===
                    if self.selected_bands is not None:
                        # Ottieni il numero di canali RIMASTI dopo il trimming
                        available_channels = chunk.shape[1] 
                        max_requested_band = self.selected_bands.max().item()
                        
                        # Check: L'indice richiesto esiste nei dati trimmati?
                        if max_requested_band >= available_channels:
                            filename = os.path.basename(file_path)
                            raise ValueError(
                                f"BANDA MANCANTE! Richiesto indice {max_requested_band}, "
                                f"ma il file '{filename}' (trimmato) ha solo {available_channels} canali."
                            )

                        # Seleziona solo gli indici specificati
                        chunk = chunk.index_select(1, self.selected_bands)

                    class_tensors.append(chunk)
                    current_count += chunk.size(0)

                except ValueError as ve:
                    # Errore critico di configurazione (es. bande sbagliate), blocchiamo tutto
                    raise ve
                except Exception as e:
                    print(f"Errore generico caricamento {file_path}: {e}")

            if not class_tensors:
                continue

            # Unisci tutti i chunk della classe
            full_class_data = torch.cat(class_tensors, dim=0)

            # === 3. APPLICA IL TAGLIO NUMERICO (CUT SAMPLES) ===
            if sample_number_per_class is not None:
                full_class_data = full_class_data[:sample_number_per_class]
            
            # Label
            labels = torch.full((full_class_data.size(0),), class_idx, dtype=torch.long)
            
            final_data_list.append(full_class_data)
            final_labels_list.append(labels)
            
            print(f"   Classe {class_idx}: Caricati {full_class_data.size(0)} campioni. Shape: {full_class_data.shape}")

        # 4. Unisci tutto nel dataset finale
        if not final_data_list:
             raise RuntimeError("Nessun dato caricato. Controlla i path e la class_map.")

        self.data = torch.cat(final_data_list, dim=0)
        self.labels = torch.cat(final_labels_list, dim=0)
        
        # Salviamo questa info pubblica per il main.py
        self.current_channels = self.data.shape[1]
        
        print(f"--- Totale Dataset: {self.data.shape[0]} campioni | Canali Input: {self.current_channels} ---")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].float(), self.labels[idx]