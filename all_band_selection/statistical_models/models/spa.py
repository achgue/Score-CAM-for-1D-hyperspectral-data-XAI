import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Nota: Non importiamo select_local_peaks qui perché SPA richiede una logica interna custom

def select_bands(X, y, n_bands=10, min_distance=5, plot=True):
    """
    implementation of SPA (Successive Projections Algorithm) with distance constraint
    
    Args:
        X (numpy array): matrix data
        y (numpy array): labels (not used - needed for API consistency)
        n_bands (int): number of bands to select
        min_distance (int): distance between selected bands (Anti-Clustering) - prevents selecting nearby bands
        plot (bool): show plot of selected bands
        
    Returns:
        list: ordered list of selected band indices
    """
    print(f"   [SPA] Avvio selezione {n_bands} bande (Dist min: {min_distance})...")

    # standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    current_X = X_scaled.copy().astype(np.float64)
    n_features = current_X.shape[1]
    
    selected_indices = []
    
    # mask to keep track of available bands (True = available, False = not available due to proximity)
    available_mask = np.ones(n_features, dtype=bool)
    
    # first band selection (max energy)
    norms = np.linalg.norm(current_X, axis=0)
    first_band = np.argmax(norms)
    selected_indices.append(first_band)
    
    # update mask for the first selected band
    start_block = max(0, first_band - min_distance)
    end_block = min(n_features, first_band + min_distance + 1)
    available_mask[start_block:end_block] = False
    
    # successive iterations
    for i in range(1, n_bands):
        # vector of the previously selected band
        last_selected_idx = selected_indices[-1]
        v = current_X[:, last_selected_idx].reshape(-1, 1)
        
        denominator = np.dot(v.T, v)
        if denominator == 0:
            print("   [SPA Error] Vettore nullo. Stop.")
            break
            
        # orthogonal projection
        projections = np.dot(v, np.dot(v.T, current_X)) / denominator
        residuals = current_X - projections
        current_X = residuals # update X with residuals
        
        # calculate norms of residuals
        resid_norms = np.linalg.norm(current_X, axis=0)
        
        # apply the mask to set nearby bands to -1
        # instead of setting to -1 only the selected bands, we also set to -1 their NEIGHBORS.
        # This way we prevent selecting nearby bands (e.g. if we select 55, we ignore until 65).
        resid_norms[~available_mask] = -1.0

        # next band is the one with max residual norm
        next_band = np.argmax(resid_norms)
        
        # if resid_norms[next_band] == -1.0, it means we cannot find any more bands at the required distance
        if resid_norms[next_band] == -1.0:
            print(f"   [SPA Warning] Impossibile trovare altre bande a distanza {min_distance}. Stop a {len(selected_indices)}.")
            break
            
        selected_indices.append(next_band)
        
        # update mask for the newly selected band
        start_block = max(0, next_band - min_distance)
        end_block = min(n_features, next_band + min_distance + 1)
        available_mask[start_block:end_block] = False

    final_bands = selected_indices
    
    # --- PLOT ---
    if plot:
        plt.figure(figsize=(10, 5))
        mean_spectrum = np.mean(X, axis=0)
        plt.plot(mean_spectrum, label='Spettro Medio', color='gray', alpha=0.5)
        plt.scatter(final_bands, mean_spectrum[final_bands], color='red', s=50, zorder=5, label='Bande SPA')
        plt.title(f"SPA Selection (Dist={min_distance})")
        plt.xlabel("Indice Banda")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return [int(b) for b in final_bands]