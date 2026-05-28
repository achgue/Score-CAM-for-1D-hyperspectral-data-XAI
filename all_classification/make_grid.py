import os
import argparse
from PIL import Image

def create_image_grid(input_folder, output_file):
    # 1. Estensioni supportate
    valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    
    # 2. Cerca e ordina le immagini nella cartella
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]
    images.sort() # Ordine alfabetico
    
    # 3. Controlli sul numero di immagini
    if len(images) < 16:
        print(f"[ERRORE] Trovate solo {len(images)} immagini nella cartella '{input_folder}'. Ne servono 16.")
        return
    elif len(images) > 16:
        print(f"[ATTENZIONE] Trovate {len(images)} immagini. Verranno usate solo le prime 16.")
        images = images[:16]
        
    print(f"Caricamento di 16 immagini da '{input_folder}'...")
    
    # 4. Carica le immagini
    loaded_images = [Image.open(os.path.join(input_folder, img)) for img in images]
    
    # 5. Prendi le dimensioni della prima immagine come riferimento
    # Se le immagini hanno dimensioni diverse, verranno ridimensionate a questa
    base_width, base_height = loaded_images[0].size
    
    # 6. Crea la tela vuota (8 colonne x 2 righe)
    grid_width = 8 * base_width
    grid_height = 2 * base_height
    
    # Crea un'immagine bianca di base (utile se ci sono trasparenze)
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    # 7. Incolla ogni immagine nella posizione corretta
    for index, img in enumerate(loaded_images):
        # Se l'immagine non è grande quanto la prima, la ridimensiona
        if img.size != (base_width, base_height):
            img = img.resize((base_width, base_height), Image.Resampling.LANCZOS)
            
        # Calcola riga (0 o 1) e colonna (da 0 a 7)
        row = index // 8
        col = index % 8
        
        # Calcola le coordinate X e Y del pixel in alto a sinistra
        x = col * base_width
        y = row * base_height
        
        # Incolla sulla tela
        grid_image.paste(img, (x, y))
        
    # 8. Salva il risultato
    grid_image.save(output_file, quality=95)
    print(f"[SUCCESSO] Griglia 2x8 salvata correttamente in: {output_file}")

if __name__ == "__main__":
    # Configurazione interfaccia a riga di comando
    parser = argparse.ArgumentParser(description="Crea una griglia 2x8 partendo da 16 immagini in una cartella.")
    parser.add_argument("-i", "--input", required=True, help="Percorso della cartella contenente le immagini")
    parser.add_argument("-o", "--output", default="griglia_risultato.jpg", help="Nome/percorso del file di output (default: griglia_risultato.jpg)")
    
    args = parser.parse_args()
    
    create_image_grid(args.input, args.output)