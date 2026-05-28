from src.dataset import UniversalSpectralDataset

def load_data_matrix(dataset_path, class_map,  max_samples=1800, trim_start=20, trim_end=20):
    print(f"Caricamento dati da {dataset_path}...")
    
    # 1. Passiamo i parametri di trim DIRETTAMENTE al Dataset
    dataset = UniversalSpectralDataset(
        root_dir=dataset_path, 
        split='train', 
        class_map=class_map, 
        sample_number_per_class=max_samples,
        bands_path=None,
        trim_start=trim_start,  # <--- Sfrutta la logica interna
        trim_end=trim_end       # <--- Sfrutta la logica interna
    )
    
    # 2. Estrazione dati (sono GIA' trimmati)
    X_tensor = dataset.data
    y_tensor = dataset.labels

    # 3. Prendi pixel centrale (1,1) -> [N, Channels]
    X_center = X_tensor[:, :, 1, 1]

    # 4. Converti in Numpy
    X_np = X_center.numpy()
    y_np = y_tensor.numpy()
    
    print(f"   Converted to Numpy: X shape {X_np.shape}, y shape {y_np.shape}")

    # Non serve fare slicing manuale qui, X_np è già pulito!
    return X_np, y_np