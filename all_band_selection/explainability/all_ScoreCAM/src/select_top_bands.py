import numpy as np
import os
import json
from scipy.signal import find_peaks

# --- CONFIGURAZIONE FILE ---
INPUT_FILE = 'global_saliencies_data.npy'
OUTPUT_FILE_UNION = 'selected_bands_peaks_union.txt'
OUTPUT_FILE_INTERSECT = 'selected_bands_peaks_intersection.txt'
OUTPUT_FILE_PER_CLASS = 'selected_bands_peaks_per_class_RANKED.txt' # Nome file aggiornato
OUTPUT_JSON = 'selected_bands_peaks_dict_ranked.json'

# --- CONFIGURAZIONE ALGORITMO ---
IGNORE_START = 5        # Ignora bande 0-4
IGNORE_END = 255        # Ignora bande da 255 in poi
MIN_PROMINENCE = 0.05   # Il picco deve svettare almeno del 5% rispetto ai vicini
MIN_HEIGHT_REL = 0.15   # Il picco deve essere alto almeno il 15% del massimo assoluto della classe
MIN_DISTANCE = 5        # Distanza minima tra due picchi

