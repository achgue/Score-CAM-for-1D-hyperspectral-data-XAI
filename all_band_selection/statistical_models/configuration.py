# config_datasets.py

# configs/selection_config.py
from models import spa, vip, rfe, wrc

# Parametri globali
SELECTION_PARAMS = {
    'n_output_bands': 2,    # Quante bande vogliamo alla fine
    'max_samples_per_class': 500, # IMPORTANTE: SPA/RFE sono lenti, non usare tutto il dataset!
    'output_dir': 'configs/selected_bands' # Dove salvare i file .txt generati
}

# --- PARAMETRI PREPROCESSING GLOBALI ---
PREPROCESSING_CONFIG = {
    'trim_start': 20,  # Taglia le prime 20 bande
    'trim_end': 20     # Taglia le ultime 20 bande
}

DATASET_CATALOG = [
    {
        'name': 'baumlein',
        'path': '../../all_datasets/output_pt/baumlein/model_train_split',
        'class_map': 
        {
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
    },
    {
        'name': 'baumpoly',
        'path': '../../all_datasets/output_pt/baumpoly/model_train_split',
        'class_map': {'Polyester': 0, 'Baumwolle': 1}
    },
    # Altri dataset...
]

# Dizionario Metodi + Parametri specifici (se servono)
MODELS = {
    'SPA': {
        'func': spa.select_bands,
        'params': {}, # SPA di solito non ha iperparametri complessi oltre n_bands
    },
#    'RFE': {
#        'func': rfe.select_bands,
#        'params': {'step': 1} # RFE specifico
#    },
    'PLS-VIP': {
        'func': vip.select_bands,
        'params': {'n_components': 5} # VIP richiede n_components
    },
    'WRC': {
        'func': wrc.select_bands,
        'params': {'n_components': 5}
    }
}