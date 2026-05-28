# --- PARAMETRI GLOBALI SCORE-CAM ---
SCORECAM_PARAMS = {
    'max_samples_per_class': 200,   # Score-CAM è LENTO. Inutile farlo su 20000 sample per classe.
    'output_dir': 'results/scorecam', # Dove salvare i .npy o .txt generati
    'device': 'cuda' # o 'cpu'
}

# --- PARAMETRI PREPROCESSING GLOBALI ---
PREPROCESSING_CONFIG = {
    'trim_start': 20,  # Taglia le prime 20 bande
    'trim_end': 20     # Taglia le ultime 20 bande
}

DATASET_CATALOG = [
    {
        'name': 'baumlein',
        'path': '../../../all_datasets/output_pt/baumlein/model_train_split',
       'class_map': {
            # --- Baumwolle (8 file) ---
            "Papier_Baumwolle_1": 0,
            "Papier_Baumwolle_2": 1,
            "Papier_Baumwolle_3": 2,
            "Papier_Baumwolle_4": 3,
            "Papier_Baumwolle_5": 4,
            "Papier_Baumwolle_6": 5,
            "Papier_Baumwolle_7": 6,
            "Papier_Baumwolle_8": 7,
            # --- Leinen (8 file) ---
            "Papier_Leinen_222": 8,
            "Papier_Leinen_269": 9,
            "Papier_Leinen_275OE_grau": 10,  
            "Papier_Leinen_275OE_Weiss": 11, 
            "Papier_Leinen_316": 12,
            "Papier_Leinen_504RL": 13,
            "Papier_Leinen_712Z": 14,
            "Papier_Leinen_816_Weiss": 15
        },
        # Dove si trova il modello addestrato per QUESTO dataset
        'checkpoint_path': '../../../all_classification/checkpoints/baumlein/SpectralCNN1D_best.pt' 
    },
    {
    'name': 'baumpoly',
     'path': '../../../all_datasets/output_pt/baumpoly/model_train_split',
      'class_map': {'Polyester': 0, 'Baumwolle': 1},
       'checkpoint_path': '../../../all_classification/checkpoints/baumpoly/SpectralCNN1D_best.pt'
    },
    {
        'name': 'tanks_uniud',
        'path': '../../../all_datasets/output_pt/tanks_uniud/model_train_split',
        # Mappa: prefisso nel nome file generato dallo split script -> etichetta numerica
        'class_map': {
            'Background': 0,  # File che iniziano con Background_
            'Metal': 1        # File che iniziano con Metal_
        },
        'checkpoint_path': '../../../all_classification/checkpoints/tanks_uniud/SpectralCNN1D_best.pt'
    }
]

# Dizionario Metodi + Parametri specifici (se servono)
# --- MODELLI E TARGET LAYER ---
# La chiave deve corrispondere al nome usato durante l'addestramento
MODELS = {
    'SpectralCNN1D': {
        # Il layer esatto a cui agganciare Score-CAM.
        # Es. Se features_conv è un nn.Sequential, il .6 potrebbe essere l'ultima ReLU.
        'target_layer_name': 'features_conv.6', 
    }
    # Puoi aggiungere 'SimpleCNN', etc.
}