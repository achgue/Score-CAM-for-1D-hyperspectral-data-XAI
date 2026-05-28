import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def select_local_peaks(scores, feature_names=None, height=None, distance=5, prominence=None, top_k=None, plot=False, title="Feature Importance"):
    """
    Trova i picchi locali in un array di punteggi (es. VIP scores, RF importance).
    
    Args:
        scores (np.array): L'array 1D dei punteggi di importanza per ogni banda.
        height (float): Soglia minima assoluta (es. VIP > 1.0).
        distance (int): Distanza minima tra due picchi (es. 5 bande). Evita i cluster (20,21,22).
        prominence (float): Quanto il picco deve "svettare" rispetto al rumore circostante.
        top_k (int): Se specificato, restituisce solo i K picchi più alti trovati.
        
    Returns:
        list: Indici delle bande selezionate (picchi).
    """
    
    # 1. Trova i picchi con le regole fisiche
    # distance=10 significa: se trovi un picco a 55, ignora tutto tra 45 e 65.
    peaks, properties = find_peaks(scores, height=height, distance=distance, prominence=prominence)
    
    # 2. Filtraggio Top-K (Opzionale)
    if top_k is not None and len(peaks) > top_k:
        # Recuperiamo le altezze dei picchi trovati
        peak_heights = scores[peaks]
        # Ordiniamo e prendiamo i migliori K
        sorted_indices = np.argsort(peak_heights)[::-1][:top_k]
        peaks = peaks[sorted_indices]
        #peaks = np.sort(peaks) # Riordiniamo gli indici in ordine crescente spettrale
        
    print(f"   [PeakPicking] Trovati {len(peaks)} picchi distinti (Dist={distance}).")
    
    # 3. Plot (Per vedere cosa stiamo facendo)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(scores, color='black', alpha=0.7, label='Importance Score')
        plt.plot(peaks, scores[peaks], "x", color='red', markersize=10, label='Selected Peaks')
        if height:
            plt.axhline(height, color='green', linestyle='--', alpha=0.5, label=f'Threshold {height}')
        plt.title(f"Peak Selection: {title}")
        plt.xlabel("Band Index")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return list(peaks)