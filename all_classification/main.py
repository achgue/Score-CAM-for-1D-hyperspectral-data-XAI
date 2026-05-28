import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as utils # Assicurati di importare questo

# Importa i tuoi moduli (assumendo la struttura precedente)
from src.dataset import UniversalSpectralDataset
from src.utils import EarlyStopping
from src.models import get_model
# Importa la config definita sopra (o incollala qui)
from configs.datasets import DATASET_CATALOG, MODELS_LIST, TRAINING_PARAMS

def plot_confusion_matrix(y_true, y_pred, class_map, save_path):
    """
    Genera e salva la confusion matrix come immagine.
    """
    # Calcola la matrice
    cm = confusion_matrix(y_true, y_pred)
    
    # Recupera i nomi delle classi nell'ordine corretto (basato sui valori 0, 1, 2...)
    # Ordiniamo il dizionario per valore per essere sicuri che l'indice 0 corrisponda alla label 0
    class_names = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Salva l'immagine
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Chiude la figura per liberare memoria
    print(f"   [INFO] Confusion Matrix salvata in: {save_path}")

def get_input_channels(dataset_path):
    """Utility per aprire un file a caso e vedere quanti canali ha (depth)"""
    import glob
    # Cerca nel train split
    sample_files = glob.glob(os.path.join(dataset_path, 'train', '*.pt'))
    if not sample_files:
        raise FileNotFoundError(f"Nessun file .pt trovato in {dataset_path}/train")
    
    sample_data = torch.load(sample_files[0])
    # sample_data shape attesa: (samples, channels, height, width) -> vogliamo il secondo elemento
    print(f"   [DEBUG] Sample data shape: {sample_data.shape[1]}")
    return sample_data.shape[1]

def train_session():
    device = TRAINING_PARAMS['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- CICLO SUI DATASET ---
    for ds_conf in DATASET_CATALOG:
        ds_name = ds_conf['name']
        ds_path = ds_conf['path']
        class_map = ds_conf['class_map']
        bands_file_path = ds_conf.get('bands_file')
        num_classes = len(class_map)
        
        print(f"\n{'='*40}")
        print(f"PROCESSING DATASET: {ds_name}")
        print(f"Path: {ds_path}")
        print(f"Classes: {class_map}")
        if bands_file_path:
            print(f"Bands Filter File: {bands_file_path}")
        else:
            print(f"Bands Filter: ALL BANDS")
        print(f"{'='*40}")
        print(f"{'='*40}")




        # 2. Caricamento Dati
        try:
            train_ds = UniversalSpectralDataset(ds_path, 'train', class_map, sample_number_per_class=1800, bands_path=bands_file_path)
            val_ds = UniversalSpectralDataset(ds_path, 'val', class_map, sample_number_per_class=600, bands_path=bands_file_path)
            test_ds = UniversalSpectralDataset(ds_path, 'test', class_map, sample_number_per_class=600, bands_path=bands_file_path)
            
            train_loader = DataLoader(train_ds, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=TRAINING_PARAMS['batch_size'], shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=TRAINING_PARAMS['batch_size'], shuffle=False)

        # --- GESTIONE ERRORI E SKIP ---
        except ValueError as ve:
            # Cattura l'errore specifico delle bande o dei dati non validi
            print(f"\n{'!'*60}")
            print(f"SKIPPING DATASET '{ds_name}' DUE TO CONFIGURATION ERROR:")
            print(f"Error details: {ve}")
            print(f"{'!'*60}\n")
            continue  # <--- QUESTO È IL COMANDO CHE PASSA AL PROSSIMO DATASET

        except FileNotFoundError as fnf:
             print(f"\n[SKIP] Dataset '{ds_name}' files not found: {fnf}\n")
             continue

        except Exception as e:
            print(f"\n[CRITICAL SKIP] Unexpected error on '{ds_name}': {e}")
            import traceback
            traceback.print_exc()
            continue

        # 1. Rilevamento automatico canali
        try:
            in_channels = train_ds[0][0].shape[0]  # Prende il numero di canali dal primo campione del train set
            print(f"-> Detected Input Channels: {in_channels}")
        except Exception as e:
            print(f"Skipping {ds_name} due to error: {e}")
            continue
        
        # --- CICLO SUI MODELLI ---
        for model_name in MODELS_LIST:
            print(f"\n   Training Model: {model_name} on {ds_name}...")
            
            # Istanzia il modello con i canali corretti per QUESTO dataset
            model = get_model(model_name, in_channels, num_classes).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=TRAINING_PARAMS['lr'])
            # mode='min': vogliamo che la loss diminuisca
            # factor=0.5: dimezza il learning rate quando si attiva
            # patience=3: aspetta 3 epoche senza miglioramenti prima di ridurre il LR
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            criterion = nn.CrossEntropyLoss()
            
            # Salva in una cartella ordinata: checkpoints/dataset_name/model_name.pt
            save_dir = os.path.join('checkpoints', ds_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_best.pt")
            
            early_stopping = EarlyStopping(patience=TRAINING_PARAMS['patience'], path=save_path)

            # Prima del loop, inizializza la migliore loss a infinito
            best_val_loss = float('inf')

            for epoch in range(TRAINING_PARAMS['epochs']):
                # Train
                model.train()
                train_loss = 0
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(X)
                    loss = criterion(out, y)
                    loss.backward()
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Val
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(device), y.to(device)
                        out = model(X)
                        v_loss = criterion(out, y)
                        val_loss += v_loss.item()
                        _, predicted = torch.max(out, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()

                avg_train = train_loss / len(train_loader)
                avg_val = val_loss / len(val_loader)
                acc = 100 * correct / total
                
                ### NUOVO: Step dello scheduler (se usi ad esempio ReduceLROnPlateau) ###
                scheduler.step(avg_val)  # Passa la loss di validazione per decidere se ridurre il LR

                print(f"   [Learning rate][lr: {optimizer.param_groups[0]['lr']:.6f}]")
                print(f"   [{model_name}][Ep {epoch+1}] T_Loss: {avg_train:.4f} | V_Loss: {avg_val:.4f} | Acc: {acc:.2f}%")

                ### NUOVO: Model Checkpointing (Salva solo se migliora!) ###
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save(model.state_dict(), save_path)
                    print(f"      -> Nuovo miglior modello salvato! (V_Loss: {best_val_loss:.4f})")
                    
                # EARLY STOPPING (Se decidi di riattivarlo)
                early_stopping(avg_val, model)
                if early_stopping.early_stop:
                   print(f"   Early stopping! Best Acc: {acc:.2f}%")
                   break

            print(f"   >>> Finished {model_name} on {ds_name}. Il miglior modello è stato salvato in {save_path}")

            # --- TEST & CONFUSION MATRIX ---
            # 1. Ricarica il peso MIGLIORE salvato dall'Early Stopping
            # (save_path è quello definito prima: checkpoints/dataset/model_best.pt)
            model.load_state_dict(torch.load(save_path))
            model.eval() 

            all_preds = []
            all_labels = []

            print(f"   Generating Confusion Matrix on Test Set...")
            
            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Sposta su CPU e converti in numpy per sklearn
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            # 2. Definisci il nome del file per la matrice
            # Es: checkpoints/baumlein/SimpleCNN_confusion_matrix.png
            cm_save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix_{in_channels}bands.png")
            
            # 3. Genera il grafico
            plot_confusion_matrix(all_labels, all_preds, class_map, cm_save_path)

if __name__ == "__main__":
    train_session()