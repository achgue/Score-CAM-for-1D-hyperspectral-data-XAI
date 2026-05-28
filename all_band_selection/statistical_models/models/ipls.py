import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from utils.peak_selector import select_local_peaks

def fit_evaluate_plsda(X, y, cv, n_components):
    """
    Helper interno: Addestra PLS-DA e calcola l'accuracy in Cross-Validation.
    """
    pls = PLSRegression(n_components=n_components)
    
    # Predizione in CV (evita overfitting)
    y_pred_continuous = cross_val_predict(pls, X, y, cv=cv, n_jobs=-1)
    
    # Conversione output regressione -> classi (0 o 1)
    # Se y è multiclasse (es. 0,1,2), questo approccio semplice (round) funziona
    # ma per robustezza usiamo argmax se dummy coded, oppure round se scalare.
    # Qui assumiamo y codificato come interi 0, 1, 2...
    y_pred_class = np.round(y_pred_continuous).astype(int)
    y_pred_class = np.clip(y_pred_class, 0, np.max(y)) # Clip per sicurezza
    
    return accuracy_score(y, y_pred_class)

def select_bands(X, y, n_bands=None, n_intervals=30, max_intervals=5, n_components=3, plot=False):
    """
    Seleziona le bande usando iPLS-DA (Interval PLS) con Forward Selection.
    
    Args:
        X, y: Dati e label
        n_bands: (Ignorato qui, usiamo max_intervals per controllare la quantità)
        n_intervals: In quanti pezzi dividere lo spettro (es. 30)
        max_intervals: Quanti intervalli selezionare al massimo
        n_components: Componenti latenti PLS
        
    Returns:
        list: Lista di TUTTE le bande incluse negli intervalli selezionati.
    """
    print(f"   [iPLS] Avvio Forward Selection (Max {max_intervals} intervalli su {n_intervals} totali)...")

    n_features = X.shape[1]
    
    # 1. Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Creazione Intervalli
    interval_size = n_features // n_intervals
    intervals = [] 
    for i in range(n_intervals):
        start = i * interval_size
        end = (i + 1) * interval_size if i < n_intervals - 1 else n_features
        intervals.append((start, end))
        
    # 3. Forward Selection Loop
    selected_interval_indices = []  # Indici degli intervalli (0..29)
    available_indices = list(range(n_intervals))
    history_acc = []
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 3-Fold per velocità
    
    for step in range(max_intervals):
        best_acc = -1
        best_candidate = -1
        
        # Prova ad aggiungere ogni intervallo disponibile
        for candidate in available_indices:
            # Costruisci indici correnti: già scelti + candidato
            current_cols = []
            
            # Aggiungi bande degli intervalli già scelti
            for idx in selected_interval_indices:
                s, e = intervals[idx]
                current_cols.extend(range(s, e))
            
            # Aggiungi bande del candidato
            s_c, e_c = intervals[candidate]
            current_cols.extend(range(s_c, e_c))
            
            # Valuta
            acc = fit_evaluate_plsda(X_scaled[:, current_cols], y, cv, n_components)
            
            if acc > best_acc:
                best_acc = acc
                best_candidate = candidate
        
        # Se l'accuratezza non migliora significativamente (o scende), potremmo fermarci.
        # Qui continuiamo fino a max_intervals per coerenza.
        if best_candidate != -1:
            selected_interval_indices.append(best_candidate)
            available_indices.remove(best_candidate)
            history_acc.append(best_acc)
            # print(f"      + Int {best_candidate} (Acc: {best_acc:.4f})")
        else:
            break

    # 4. Costruzione Output Finale
    # Convertiamo gli indici degli intervalli in indici delle bande vere e proprie
    final_bands = []
    for idx in selected_interval_indices:
        start, end = intervals[idx]
        final_bands.extend(range(start, end))
        
    final_bands = sorted(list(set(final_bands)))
    
    # --- PLOT (Opzionale) ---
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(history_acc)+1), history_acc, 'o-')
        plt.title(f"iPLS Forward Selection (Final Acc: {history_acc[-1]:.2%})")
        plt.xlabel("Step")
        plt.ylabel("CV Accuracy")
        plt.grid(True)
        plt.show()

    return [int(b) for b in final_bands]