import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MODELLO 1: CNN 2D (Ottimizzata per patch piccoli) ---
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Conv1: Mantiene 3x3 (padding=1, kernel=3)
        # Nota: input_channels viene passato dinamicamente dal main (es. 256 o 512)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Conv2: Mantiene 3x3
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Conv3: Opzionale, per estrarre più features
        #self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm2d(256)

        # Global Average Pooling: 
        # Riduce qualsiasi dimensione (H x W) a (1 x 1).
        # Questo risolve il problema del calcolo delle dimensioni per il Linear layer.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Il layer lineare prende in ingresso l'ultimo numero di canali (128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [Batch, Channels, 3, 3]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x) # Output shape: [Batch, 256, 1, 1]
        x = x.view(x.size(0), -1) # Flatten: [Batch, 256]
        x = self.fc(x)
        return x

# --- MODELLO 2: MLP (Fully Connected) ---
class MLP(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MLP, self).__init__()
        # Input: input_channels * 3 * 3 (appiattiamo tutto il cubo 3x3)
        self.input_dim = input_channels * 3 * 3
        
        self.layer1 = nn.Linear(self.input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4)
        
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        # Flatten dell'input: da [Batch, C, 3, 3] a [Batch, C*9]
        x = x.view(x.size(0), -1)
        #print(f"   [DEBUG] MLP input: {x}")
        #print(f"   [DEBUG] MLP input shape after flatten: {x.shape}")
        
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.out(x)
        return x

# --- MODELLO 3: 1D CNN (Spettrale) ---
# Tratta il pixel centrale (o la media del patch) come una sequenza
class SpectralCNN1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SpectralCNN1D, self).__init__()
        
        # --- BLOCCO CONVOLUZIONALE (Feature Extraction) ---
        # Conv1: Input [Batch, 1, Channels] -> Output [Batch, 64, Channels]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2) # Dimezza la lunghezza dello spettro
        self.relu = nn.ReLU()
        
        # Conv2: Input [Batch, 64, Channels/2] -> Output [Batch, 128, Channels/2]
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Raggruppiamo tutto nel Sequential come richiesto
        self.features_conv = nn.Sequential(
            self.conv1,
            self.bn1,
            self.maxpool1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu
        )

        # --- BLOCCO CLASSIFICATORE (Dense Head) ---
        
        # GLOBAL POOLING: Fondamentale per la "universalità".
        # Indipendentemente se lo spettro in input era lungo 384 o 26000,
        # questo layer riduce la dimensione temporale a 1.
        # Output shape: [Batch, 128, 1]
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        
        self.flatten = nn.Flatten()
        
        # Ora l'input della Linear è sempre 128 (i canali di uscita della conv2),
        # non dipende più dalla lunghezza dello spettro originale.
        self.dense1 = nn.Linear(128, 256) 
        self.bn3 = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, num_classes)
        
        # Nota: Non definiamo Softmax qui perché nn.CrossEntropyLoss lo include già internamente durante il training.

    def forward(self, x):
        # x shape in input: [Batch, Channels, 3, 3]
        
        # 1. ESTRAZIONE PIXEL CENTRALE
        # Prendiamo solo il pixel alle coordinate (1, 1) della patch 3x3
        x = x[:, :, 1, 1]  # Output shape: [Batch, Channels]
        
        # 2. Reshape per Conv1D
        # La Conv1D vuole [Batch, Canali_Ingresso, Lunghezza_Sequenza]
        # Per noi lo spettro è la sequenza, e abbiamo 1 solo "canale" di intensità.
        x = x.unsqueeze(1) # Output shape: [Batch, 1, Channels]
        
        # 3. Feature Extraction
        x = self.features_conv(x)
        
        # 4. Classification Head (come da tua struttura)
        x = self.maxpool2(x)
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.dense2(x)
        
        # In fase di training restituiamo i logits grezzi.
        return x

# --- FACTORY ---
def get_model(model_name, input_channels, num_classes):
    if model_name == 'SimpleCNN':
        return SimpleCNN(input_channels, num_classes)
    elif model_name == 'MLP':
        return MLP(input_channels, num_classes)
    elif model_name == 'SpectralCNN1D':
        return SpectralCNN1D(input_channels, num_classes)
    else:
        raise ValueError(f"Model {model_name} not implemented")