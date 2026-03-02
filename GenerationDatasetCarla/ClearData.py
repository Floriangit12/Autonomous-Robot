import os
import glob
import numpy as np

# 1. Le chemin vers le dossier qui contient TOUS tes .npz
DATASET_DIR = r"C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\finalDataset\points"
IMAGES_DIR  = r"C:\Users\flori\Desktop\QAT_TEST_2026\GenerateDataSetCarla\DataSet\finalDataset\images"

# ⚠️ Mettre à True pour SUPPRIMER les fichiers (.npz + images) des frames filtrées
DELETE_BAD_FRAMES = False

CAM_NAMES_DELETE = ["front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right"]

# 2. On récupère la liste de tous les fichiers .npz dans ce dossier
npz_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
npz_files.sort()

print(f"🔍 Analyse de {len(npz_files)} fichiers en cours...")
if DELETE_BAD_FRAMES:
    print("🗑️  Mode SUPPRESSION activé !\n")
else:
    print("👀 Mode lecture seule (DELETE_BAD_FRAMES = False)\n")

frames_trouvees = 0
files_deleted = 0

# 3. On boucle sur chaque fichier
for file_path in npz_files:
    try:
        data = np.load(file_path, allow_pickle=True)
        labels = data["labels"].copy()  # copie en mémoire
        points = data["points"].copy()  # copie en mémoire
        data.close()  # libère le verrou sur le fichier
        unique_labels = np.unique(labels)
        # 4. On vérifie si la frame contient 3 classes ou moins
        if len(unique_labels) <= 3 or points.shape[0] <= 20000:
            frame_name = os.path.basename(file_path)
            stem = frame_name.replace(".npz", "")  # ex: "frame_000000"
            print(f"⚠️ {frame_name} | Nb de classes: {len(unique_labels)} | Classes présentes : {unique_labels}")
            frames_trouvees += 1

            if DELETE_BAD_FRAMES:
                # Suppression du .npz
                os.remove(file_path)
                files_deleted += 1
                print(f"   🗑️  Supprimé : {frame_name}")

                # Suppression des images associées (toutes les caméras)
                for cam in CAM_NAMES_DELETE:
                    for ext in (".jpg", ".png"):
                        img_path = os.path.join(IMAGES_DIR, f"{stem}_{cam}{ext}")
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            files_deleted += 1
                            print(f"   🗑️  Supprimé : {os.path.basename(img_path)}")

    except Exception as e:
        print(f"❌ Erreur avec le fichier {os.path.basename(file_path)} : {e}")

print(f"\n✅ Analyse terminée ! {frames_trouvees} frame(s) trouvée(s) avec 3 classes ou moins.")
if DELETE_BAD_FRAMES:
    print(f"🗑️  Total fichiers supprimés : {files_deleted}")