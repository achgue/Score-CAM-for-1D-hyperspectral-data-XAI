import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from utils.peak_selector import select_local_peaks

def select_bands(X, y, n_bands=10, n_components=5, plot=False):
    """
    selects bands using Weighted Regression Coefficients (WRC) from PLS
    using peak-picking to avoid collinearity.
    
    Args:
        X (numpy array): matrix of data (N_samples, N_bands)
        y (numpy array): vector of labels (N_samples,)
        n_bands (int): number of bands to select
        n_components (int): latent components for PLS
        plot (bool): if True, shows the peak graph using peak_selector
        
    Returns:
        list: ordered list of indices of selected bands
    """
    
    # pls needs data to be standardized because the coefficients depend on the scale of the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # training PLS
    print(f"   [WRC] Training PLS con {n_components} componenti...")
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y)

    # pls.coef_ is shape (n_features, n_targets). We flatten to 1D.
    coefficients = np.array(pls.coef_).flatten()
    
    # pick the absolute value: we care about "how much" it weighs, not the sign
    wrc_scores = np.abs(coefficients)
    
    # pick peaks using the custom function to find local maxima separated by a certain distance
    selected_indices = select_local_peaks(
        scores=wrc_scores,
        height=None,       # no fixed minimum height, we trust the top_k to filter
        distance=5,       # avoids selecting adjacent bands (e.g. if it picks 55, it ignores until 65)
        top_k=n_bands,     # pick only the top N peaks
        plot=plot,         
        title=f"WRC (PLS Beta Coefficients) - Top {n_bands}"
    )

    return selected_indices