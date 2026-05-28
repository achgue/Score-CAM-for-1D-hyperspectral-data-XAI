import os
import torch
from torch.utils.data import Dataset
import glob

class UniversalSpectralDataset(Dataset):
    def __init__(self, root_dir, split='train', class_map=None, sample_number_per_class=1800, bands_path=None):
        """
        sample_number_per_class: int, quanti campioni prendere ESATTAMENTE per ogni classe (se disponibili).
        """
        self.class_map = class_map

        # --- LOGICA SELEZIONE BANDE ---
        self.selected_bands = None
        if bands_path and os.path.exists(bands_path):
            print(f"   [Info] Reading bands from {bands_path}...")
            try:
                with open(bands_path, 'r') as f:
                    # Legge le linee, rimuove spazi vuoti, converte in int
                    indices = [int(line.strip()) for line in f if line.strip().isdigit()]
                    indices = sorted(set(indices))  # Rimuove duplicati e ordina
                
                if indices:
                    # Converte in LongTensor per usare come indice avanzato in PyTorch
                    self.selected_bands = torch.tensor(indices, dtype=torch.long)
                    print(f"   [Info] Selected {len(self.selected_bands)} bands.")
                else:
                    print("   [Warning] Bands file is empty. Using ALL bands.")
            except Exception as e:
                print(f"   [Error] Could not read bands file: {e}. Using ALL bands.")
        
        # --- CARICAMENTO FILE ---
        search_path = os.path.join(root_dir, split, '*.pt')
        all_files = glob.glob(search_path)
        
        if not all_files:
            raise FileNotFoundError(f"Nessun file trovato in {search_path}")

        print(f"--- Caricamento Dataset: {split} ---")
        print(f"Target: {sample_number_per_class} campioni per classe.")

        # 1. Organizza i file per classe
        # files_by_class = {0: ['file1.pt', 'file2.pt'], 1: ['file3.pt'], ...}
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

        # 2. Itera su ogni classe per caricare e tagliare i dati
        for class_idx, file_list in files_by_class.items():
            if not file_list:
                print(f"   Classe {class_idx}: Nessun file trovato.")
                continue

            # Accumulatore temporaneo per QUESTA classe
            class_tensors = []
            current_count = 0
            
            for file_path in file_list:
                # Se abbiamo già raggiunto il target per questa classe, smettiamo di aprire file
                if sample_number_per_class is not None and current_count >= sample_number_per_class:
                    break
                
                try:
                    # Carica il blocco (es. [20000, 384, 3, 3])
                    chunk = torch.load(file_path)
                    # --- APPLICA IL FILTRO BANDE QUI (Risparmia Memoria) ---
                    if self.selected_bands is not None:
                        # Ottieni il numero di canali nel file corrente
                        available_channels = chunk.shape[1] 
                        max_requested_band = self.selected_bands.max().item()
                        
                        # SE L'INDICE È FUORI RANGE -> SCATENA ERRORE
                        if max_requested_band >= available_channels:
                            raise ValueError(
                                f"BANDA MANCANTE! Richiesto indice {max_requested_band}, "
                                f"ma il file '{filename}' ha solo {available_channels} canali."
                            )

                        # Seleziona solo gli indici specificati sulla dim 1 (Canali)
                        chunk = chunk.index_select(1, self.selected_bands)
                    class_tensors.append(chunk)
                    current_count += chunk.size(0)
                except ValueError as ve:
                    # Rilanciamo l'errore critico per bloccare l'init e avvisare il main
                    raise ve
                except Exception as e:
                    print(f"Errore caricamento {file_path}: {e}")

            if not class_tensors:
                continue

            # Unisci tutti i chunk della classe in un unico tensore
            full_class_data = torch.cat(class_tensors, dim=0)

            
            # 3. APPLICA IL TAGLIO (CUT)
            # Se sample_number_per_class è definito, prendiamo solo i primi N
            if sample_number_per_class is not None:
                full_class_data = full_class_data[:sample_number_per_class]
            
            # Creiamo le label corrispondenti
            num_actual_samples = full_class_data.size(0)
            labels = torch.full((num_actual_samples,), class_idx, dtype=torch.long)
            
            print(f"   Classe {class_idx}: Caricati {num_actual_samples} campioni (Target: {sample_number_per_class} - Dimensione finale: {full_class_data.shape})")

            final_data_list.append(full_class_data)
            final_labels_list.append(labels)

        # 4. Unisci tutto nel dataset finale
        if not final_data_list:
             raise RuntimeError("Nessun dato caricato. Controlla i path e la class_map.")

        self.data = torch.cat(final_data_list, dim=0)
        self.labels = torch.cat(final_labels_list, dim=0)
        
        print(f"--- Totale Dataset: {self.data.shape[0]} campioni ---")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Restituisce il singolo campione (C, H, W) e la label
        return self.data[idx].float(), self.labels[idx]