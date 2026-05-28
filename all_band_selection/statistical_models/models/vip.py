import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from utils.peak_selector import select_local_peaks

def calculate_vip(model):
    """
    Calcola i punteggi VIP (Variable Importance in Projection) per un modello PLS.
    Helper function interna.
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    
    n_features = w.shape[0]
    n_components = w.shape[1]
    
    vips = np.zeros((n_features,))
    
    # Calcola la varianza spiegata per ogni componente
    s = np.diag(np.dot(t.T, t).dot(np.dot(q.T, q)))
    total_s = np.sum(s)
    
    for i in range(n_features):
        weight_sum = 0
        for k in range(n_components):
            weight_k = (w[i, k] / np.linalg.norm(w[:, k]))**2
            weight_sum += s[k] * weight_k
            
        vips[i] = np.sqrt(n_features * weight_sum / total_s)
        
    return vips

def select_bands(X, y, n_bands=10, n_components=5, plot=False):
    """
    select bands using VIP analysis on PLS.
    
    Args:
        X (numpy array): matrix of data (N_samples, N_bands)
        y (numpy array): vector of labels (N_samples,)  
        n_bands (int): number of bands to select (the top N by VIP score)
        n_components (int): PLS components
        plot (bool): show the plot
        
    Returns:
        list: ordered list of indices of selected bands
    """
    
    # vip requires centered and scaled data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # training PLS
    print(f"   [VIP] Training PLS con {n_components} componenti...")
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y)
    
    # calculate VIP Scores
    vip_scores = calculate_vip(pls)
    
    # pick peaks using the custom function to find local maxima separated by a certain distance
    selected_indices = select_local_peaks(
        scores=vip_scores, 
        height=0,      # Regola aurea VIP > 1
        distance=5,     # <--- QUESTA È LA MAGIA CHE RIMUOVE I CLUSTER (20, 21, 22)
        top_k=n_bands,   # Prendi al massimo n_bands
        plot=plot,
        title="PLS-VIP Peaks"
    )

    # --- PLOT ---
    if plot:
        plt.figure(figsize=(10, 5))
        plt.axhline(y=1, color='r', linestyle='--', label='Soglia VIP > 1')
        plt.plot(vip_scores, label='VIP Score', color='black')
        # Evidenzia le bande selezionate
        plt.scatter(top_indices, vip_scores[top_indices], color='green', zorder=5, label='Selected')
        plt.title(f"VIP Selection (Top {n_bands})")
        plt.xlabel("Spectral Band Index")
        plt.ylabel("VIP Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 5. Return lista pulita
    return selected_indices