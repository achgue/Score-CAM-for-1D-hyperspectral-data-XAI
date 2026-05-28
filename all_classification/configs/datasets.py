# config_datasets.py
DATASET_CATALOG = [
    
#    {
#        'name': 'baumpoly',
#        'path': '../all_datasets/output_pt/baumpoly/model_train_split',
#        # Mappa: prefisso nel nome file generato dallo split script -> etichetta numerica
#        'class_map': {
#            'Polyester': 0,  
#            'Baumwolle': 1   
#        },
#        'bands_file': 'configs/bands_baumpoly.txt', 
#        'class_colors': {
#            0: "#1fb4a8",  # Blu (Poliestere)
#            1: "#db4590"   # Rosso (Cotone)
#        }
#    },
#    
#    {
#        'name': 'baumlein',
#        'path': '../all_datasets/output_pt/baumlein/model_train_split',
#        # Mappa: "testo nel nome file" -> etichetta numerica
#        'class_map': {
#            # --- Baumwolle (8 file) ---
#            "Papier_Baumwolle_1": 0,
#            "Papier_Baumwolle_2": 1,
#            "Papier_Baumwolle_3": 2,
#            "Papier_Baumwolle_4": 3,
#            "Papier_Baumwolle_5": 4,
#            "Papier_Baumwolle_6": 5,
#            "Papier_Baumwolle_7": 6,
#            "Papier_Baumwolle_8": 7,
#            
#            # --- Leinen (8 file) ---
#            "Papier_Leinen_222": 8,
#            "Papier_Leinen_269": 9,
#            "Papier_Leinen_275OE_grau": 10,  
#            "Papier_Leinen_275OE_Weiss": 11, 
#            "Papier_Leinen_316": 12,
#            "Papier_Leinen_504RL": 13,
#            "Papier_Leinen_712Z": 14,
#            "Papier_Leinen_816_Weiss": 15
#        },
#        'class_colors': {
#            # --- BAUMWOLLE (Schema Caldo Neon: Fucsia/Magenta -> Giallo) ---
#            0: '#FF0080',  # Fucsia Neon
#            1: '#FF004D',  # Rosa Lampone
#            2: '#FF3333',  # Rosso Brillante
#            3: '#FF6600',  # Arancione Acceso
#            4: '#FF9900',  # Arancione Dorato
#            5: '#FFCC00',  # Giallo Zafferano
#            6: '#FFE600',  # Giallo Limone
#            7: '#FFFF66',  # Giallo Pastello Luminoso
#            
#            # --- LEINEN (Schema Freddo Neon: Verde/Verde Acqua -> Viola) ---
#            8: '#00FF87',  # Verde Primavera / Menta Neon
#            9: '#00F5D4',  # Verde Acqua / Teal
#            10: '#00E5FF', # Ciano / Azzurro Neon
#            11: '#0099FF', # Azzurro Intenso
#            12: '#5E60CE', # Indaco Luminoso
#            13: '#8040F0', # Viola Elettrico
#            14: '#B366FF', # Viola Ametista
#            15: '#D999FF', # Lilla Luminoso
#        },
#        'bands_file': 'configs/bands_baumlein.txt'
#    },

    {
        'name': 'tanks_uniud',
        'path': '../all_datasets/output_pt/tanks_uniud/model_train_split',
        # Mappa: prefisso nel nome file generato dallo split script -> etichetta numerica
        'class_map': {
            'Background': 0,  
            'Metal': 1        
        },
        'bands_file': 'configs/bands_tanks_uniud.txt', 
        'class_colors': {
            0: '#404040',  # Grigio scuro (Asfalto/Vegetazione/Sfondo)
            1: '#FFD700'   # Giallo Oro / Arancione (Metallo) - molto visibile per contrasto
        }
    }
    
]

MODELS_LIST = ['MLP'] #, 'MLP', 'ResNet']
TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'patience': 9,
    'device': 'cuda'
}

