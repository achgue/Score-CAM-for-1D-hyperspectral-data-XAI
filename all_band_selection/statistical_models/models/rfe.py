import numpy as np
import time
#from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from utils.peak_selector import select_local_peaks

def select_bands(X, y, n_bands=5, max_samples=10000, step=0.05, plot=True):
    """
    Select bands using Random Forest Freature Picking (RFFP)
    
    Args:
        X (numpy array): matrix data
        y (numpy array): labels vector
        n_bands (int): number of bands to select
        max_samples (int): maximum number of samples to use for speed (todo VALUTARE SE TOGLIERLO)
        step (float): percentage of features to remove at each iteration (0.05 = 5%) - todo VALUTARE SE TOGLIERLO, visto che non usiamo più RFE loop
        
    Returns:
        list: list of selected band indices
    """
    
    # todo QUESTA PARTE PROBABILMENTE NON SERVE PIÙ, VISTO CHE NON USIAMO PIÙ RFE LOOP
    # RFE è lento O(N*M), quindi riduciamo N se è troppo grande
    n_samples = X.shape[0]
    
    if n_samples > max_samples:
        print(f"   [RFE] Subsampling: utilizzo {max_samples} campioni casuali su {n_samples} per velocità.")
        # Generiamo indici casuali
        indices = np.random.choice(n_samples, max_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        X_subset = X
        y_subset = y

    print(f"   [RFE] Avvio selezione di {n_bands} bande (Step={step})...")
    start_time = time.time()

    # estimator configuration
    # - n_estimators=100: number of trees in the forest. More trees can give better performance but are slower. 100 is a common default
    # - n_jobs=-1: use all CPU cores for parallel processing
    # - random_state=42: seed for reproducibility
    estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    estimator.fit(X_subset, y_subset)
    
    # extract feature importances from the fitted Random Forest
    importances = estimator.feature_importances_
    
    end_time = time.time()
    print(f"   [RFE] Completato in {end_time - start_time:.1f} secondi.")
    
    # featre picking with peak picking to avoid selecting nearby bands (anti-clustering)
    selected_indices = select_local_peaks(
        scores = importances,           # use the feature importances as scores for peak picking
        height=np.mean(importances),    # edit to increase the number of selected bands (since RF importances are often low, we set the threshold to the mean)
        distance=5,                    # prevent selecting nearby bands (e.g. if we select 55, ignore until 65)
        top_k=n_bands,
        plot=plot,
    )
    
    return selected_indices