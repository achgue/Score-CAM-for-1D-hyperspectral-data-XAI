# Pipeline completa per Tank Uniud

Questo documento descrive la pipeline end-to-end usata in questa cartella `code` per:

1. preparare il dataset iperspettrale `tanks_uniud`,
2. addestrare un classificatore CNN 1D / MLP nella cartella `all_classification`,
3. estrarre le bande più informative tramite Score-CAM nella cartella `all_band_selection/explainability/all_ScoreCAM`.

---

## 0. Fase preliminare

Prima di eseguire qualsiasi linea di codice:

- aprire il terminale con wsl e posizionarsi sulla root folder del progetto
- installare l'ambiente tramite il seguente comando

```bash
conda env create -f environment.yml
```

- attivare l'ambiente tramite

```bash
conda activate achgue_unmixdiff
```

A questo punto si può procedere con la pipe line. Se dovessero esservi eventuali problematiche a livello di dipendenze, posizionatevi sulla root e installate i requirements tramite

```bash
pip install -r libraries_to_install.txt
```

---

## 1. Obiettivo della pipeline

La pipeline prende i dati raw HSI di `tanks_uniud` e li trasforma in:

- patch iperspettrali pronte per il training,
- split train/val/test,
- un classificatore addestrato,
- una stima delle bande rilevanti tramite explainability (Score-CAM).

L’ordine logico è:

`all_datasets` -> `all_classification` -> `all_band_selection/explainability/all_ScoreCAM`

---

## 2. Fase 1 — Preparazione del dataset (`all_datasets`)

### Dove si trovano i dati

- Input raw:
  - `all_datasets/raw/tanks_uniud/images/`
  - `all_datasets/raw/tanks_uniud/masks/`
- Output generato:
  - `all_datasets/output_pt/tanks_uniud/patches_split/`
  - `all_datasets/output_pt/tanks_uniud/model_train_split/`

### Script usato

Nel progetto attuale, il punto di ingresso principale per il dataset `tanks_uniud` è:

- `all_datasets/main2.py`

Questo script chiama:

- `process_hsi_envi(...)` da `all_datasets/process_hsi.py`
- `split_dataset(...)` da `all_datasets/split_dataset.py`

### Cosa fa

1. legge tutte le immagini `.hdr` / `.tif` in `raw/tanks_uniud/images/` (*dataset di Monte*),
2. cerca le maschere corrispondenti in `raw/tanks_uniud/masks/`,
3. estrai patch 3x3 da ogni pixel valido,
4. salva i patch in `output_pt/tanks_uniud/patches_split/`,
5. divide i patch in `train/`, `val/`, `test/` dentro `output_pt/tanks_uniud/model_train_split/`.

### Comando rapido

Da `code/all_datasets`:

```bash
python main2.py
```

> Nota: in `main.py` la voce `tanks_uniud` è commentata, quindi per questo dataset il file corretto da usare è `main2.py`.

### Output atteso

Dopo l’esecuzione dovresti trovare:

- `all_datasets/output_pt/tanks_uniud/patches_split/...`
- `all_datasets/output_pt/tanks_uniud/model_train_split/train/*.pt`
- `all_datasets/output_pt/tanks_uniud/model_train_split/val/*.pt`
- `all_datasets/output_pt/tanks_uniud/model_train_split/test/*.pt`

---

## 3. Fase 2 — Addestramento del classificatore (`all_classification`)

### Dove si trova la configurazione

- `all_classification/main.py`
- `all_classification/configs/datasets.py`
- `all_classification/src/dataset.py`
- `all_classification/src/models.py`

### Dataset usato

Il catalogo contiene la voce:

- `tanks_uniud`

con mappa classi:

- `Background = 0`
- `Metal = 1`

Il path usato per i dati è:

- `../all_datasets/output_pt/tanks_uniud/model_train_split`

### Modello usato

Il training attuale in `all_classification/main.py` usa come default:

- `MODELS_LIST = ['SpectralCNN1D']`

Il modello `SpectralCNN1D` è implementato in `all_classification/src/models.py` (insieme ad altri modelli).

### Cosa fa lo script

1. carica `train/`, `val/`, `test/` dal dataset splittato,
2. applica eventuale filtro bande tramite `configs/bands_tanks_uniud.txt`,
3. addestra il modello con batch size, learning rate e early stopping configurati,
4. salva il miglior checkpoint in:
   - `all_classification/checkpoints/tanks_uniud/SpectralCNN1D_best.pt`
5. genera la confusion matrix sul test set.

### Comando rapido

Da `code/all_classification`:

```bash
python main.py
```

### Output atteso

- `all_classification/checkpoints/tanks_uniud/SpectralCNN1D_best.pt`
- `all_classification/checkpoints/tanks_uniud/SpectralCNN1D_confusion_matrix_...png`

### Nota per Score-CAM

Il modulo `all_band_selection/explainability/all_ScoreCAM/configuration.py` punta a un checkpoint atteso del tipo:

- `../../../all_classification/checkpoints/tanks_uniud/SpectralCNN1D_best.pt`

---

## 4. Fase 3 — Estrazione delle bande con Score-CAM (`all_band_selection/explainability/all_ScoreCAM`)

### Script usato

- `all_band_selection/explainability/all_ScoreCAM/main.py`
- `all_band_selection/explainability/all_ScoreCAM/configuration.py`

### Cosa fa

1. carica il dataset `tanks_uniud` dal test split di `all_datasets/output_pt/tanks_uniud/model_train_split`,
2. carica il checkpoint del classificatore,
3. calcola le importanze spettrali tramite Score-CAM 1D,
4. aggrega i risultati per classe,
5. salva i file in `results/scorecam/`,
6. genera bande selezionate (`union.txt`, `intersection.txt`, `peaks_per_class.txt`, `peaks_dict_ranked.json`).

### Comando rapido

Da `code/all_band_selection/explainability/all_ScoreCAM`:

```bash
python main.py
```

### Output atteso

I risultati vengono salvati in:

- `all_band_selection/explainability/all_ScoreCAM/results/scorecam/`

In particolare, per il dataset `tanks_uniud` il percorso atteso è del tipo:

- `results/scorecam/SpectralCNN1D/tanks_uniud/saliency_scores.npy`
- `results/scorecam/SpectralCNN1D/tanks_uniud/selected_bands/`

---

## 5. Flusso completo consigliato

### Variante rapida (come è strutturato oggi)

1. `cd all_datasets`
2. `python main2.py`
3. `cd ../all_classification`
4. `python main.py`
5. `cd ../all_band_selection/explainability/all_ScoreCAM`
6. `python main.py`

---

## 6. Risultato finale desiderato

Alla fine della pipeline dovresti avere:

- dataset pronto e splittato per `tanks_uniud`,
- modello addestrato per classificazione,
- saliency maps e bande selezionate ottenute con Score-CAM.
