import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mrmr import mrmr_classif

# il codice dovrebbe essere rivisto in quanto non seleziona la banda di un picco ma la prima di un possibile cluster
# in questo senso se mRMR seleziona 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 (10 bande vicine), poi seleziona solo la prima e ingnora le altre
# noi invece vorremo selezionasse quella con contenuto più informativo (es. 23) e ignorasse le vicine (20, 21, 22, 24, 25, 26, 27, 28, 29)

def select_bands(X, y, n_bands=10, max_samples=2000, min_distance=10, plot=False):
    """
    Seleziona le bande usando mRMR con post-processing per evitare cluster.
    
    Args:
        X (numpy array): Dati
        y (numpy array): Label
        n_bands (int): Numero finale di bande desiderate
        max_samples (int): Limite campioni per velocità
        min_distance (int): Distanza minima tra le bande selezionate
        plot (bool): Mostra grafico
        
    Returns:
        list: Lista ordinata degli indici delle bande selezionate
    """
    print(f"   [mRMR] Avvio selezione (Target={n_bands}, Dist={min_distance})...")
    
    # 1. Subsampling (mRMR è lento!)
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X_subset = X[idx]
        y_subset = y[idx]
    else:
        X_subset = X
        y_subset = y
    
    # 2. Preparazione DataFrame
    feature_names = [f"B_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X_subset, columns=feature_names)
    y_series = pd.Series(y_subset)
    
    # 3. Esecuzione mRMR "Abbondante"
    # TRUCCO: Chiediamo a mRMR molte più bande di quelle che ci servono (es. 2x o 3x).
    # Perché? Perché poi filtreremo via i vicini, quindi ne perderemo molte.
    k_oversampled = n_bands * 3 
    
    try:
        # show_progress=False pulisce l'output
        selected_features_raw = mrmr_classif(X=X_df, y=y_series, K=k_oversampled, show_progress=False)
    except Exception as e:
        print(f"      [Errore mRMR] {e}")
        return []

    # 4. Parsing e Filtraggio (Anti-Cluster)
    # Convertiamo subito in interi preservando l'ordine (che è l'ordine di importanza!)
    candidates = [int(f.split('_')[1]) for f in selected_features_raw]
    
    final_selection = []
    
    print(f"      [Post-Processing] Filtraggio candidati da {len(candidates)} a {n_bands}...")
    
    for candidate in candidates:
        # Se abbiamo già raggiunto il numero target, fermati
        if len(final_selection) >= n_bands:
            break
            
        # Controlla se il candidato è troppo vicino a qualcuno già selezionato
        is_too_close = False
        for selected in final_selection:
            if abs(candidate - selected) < min_distance:
                is_too_close = True
                break
        
        # Se è "lontano" da tutti, aggiungilo
        if not is_too_close:
            final_selection.append(candidate)
            
    # Ordiniamo gli indici finali per l'output (non per importanza, ma spettrale)
    final_selection = sorted(final_selection)

    # --- PLOT (Opzionale) ---
    if plot:
        plt.figure(figsize=(10, 5))
        mean_spectrum = np.mean(X, axis=0)
        plt.plot(mean_spectrum, color='gray', alpha=0.5, label='Mean Spectrum')
        
        # Evidenzia le bande
        plt.scatter(final_selection, mean_spectrum[final_selection], color='red', s=50, zorder=5, label='Selected (Filtered)')
        
        plt.title(f"mRMR Selection (Dist={min_distance})")
        plt.xlabel("Band Index")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return final_selection