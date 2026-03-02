import os
import re
import glob
import shutil
import random
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

EXPECTED_CAMERAS = [
    "front", "back", "left", "right",
    "front_left", "front_right", "back_left", "back_right"
]


def validate_frame(dataset_dir, frame_id):
    """
    Vérifie qu'une frame possède bien :
      - 1 fichier .npz dans 'points/'
      - 8 images (une par caméra) dans 'images/'
    Retourne (True, []) si tout est OK, sinon (False, [liste de problèmes]).
    """
    problems = []

    # Vérifier le fichier npz
    npz_path = os.path.join(dataset_dir, "points", f"frame_{frame_id}.npz")
    if not os.path.isfile(npz_path):
        problems.append(f"NPZ manquant : {npz_path}")

    # Vérifier les 8 images
    images_dir = os.path.join(dataset_dir, "images")
    for cam in EXPECTED_CAMERAS:
        img_path = os.path.join(images_dir, f"frame_{frame_id}_{cam}.jpg")
        if not os.path.isfile(img_path):
            problems.append(f"Image manquante : frame_{frame_id}_{cam}.jpg")

    return (len(problems) == 0, problems)


def collect_samples(input_dirs):
    """
    Récupère toutes les frames disponibles dans chaque dataset
    en se basant sur les fichiers .npz du dossier 'points'.
    Vérifie que chaque frame contient bien 8 images + 1 npz.
    Retourne une liste de tuples (dataset_dir, frame_id_str).
    """
    pattern_npz = re.compile(r"frame_(\d+)\.npz$")
    samples = []
    skipped = 0

    for d in input_dirs:
        points_dir = os.path.join(d, "points")
        if not os.path.isdir(points_dir):
            print(f"[WARN] Pas de dossier 'points' dans {d}, ignoré.")
            continue

        # os.scandir est plus rapide que os.listdir (évite des appels stat supplémentaires)
        entries = sorted(
            (e.name for e in os.scandir(points_dir) if e.is_file(follow_symlinks=False)),
        )
        for fname in entries:
            m = pattern_npz.match(fname)
            if m:
                frame_id = m.group(1)  # ex: "000012"
                valid, problems = validate_frame(d, frame_id)
                if valid:
                    samples.append((d, frame_id))
                else:
                    skipped += 1
                    print(f"[SKIP] Frame {frame_id} dans {d} :")
                    for p in problems:
                        print(f"       - {p}")

    if skipped > 0:
        print(f"\n[INFO] {skipped} frame(s) ignorée(s) car incomplète(s).")

    return samples


def _copy_single_frame(args):
    """
    Copie une frame (1 npz + ses images) — fonction exécutée par le thread pool.
    Retourne (new_idx, True/False, warnings).
    """
    new_idx, dataset_dir, old_frame_id, images_out, points_out = args
    warnings = []
    new_frame_id = f"{new_idx:06d}"

    # ---- Copier le fichier points (.npz)
    src_points = os.path.join(dataset_dir, "points", f"frame_{old_frame_id}.npz")
    dst_points = os.path.join(points_out, f"frame_{new_frame_id}.npz")

    if not os.path.isfile(src_points):
        warnings.append(f"[WARN] Points manquant : {src_points}, ignoré.")
        return (new_idx, False, warnings)

    shutil.copy2(src_points, dst_points)

    # ---- Copier toutes les images associées à cette frame
    src_images_dir = os.path.join(dataset_dir, "images")
    pattern = os.path.join(src_images_dir, f"frame_{old_frame_id}_*.jpg")
    img_paths = glob.glob(pattern)

    if not img_paths:
        warnings.append(
            f"[WARN] Aucune image trouvée pour frame {old_frame_id} dans {dataset_dir}."
        )

    for src_img in img_paths:
        suffix = src_img.split(f"frame_{old_frame_id}", 1)[1]
        dst_img = os.path.join(images_out, f"frame_{new_frame_id}{suffix}")
        shutil.copy2(src_img, dst_img)

    return (new_idx, True, warnings)


def merge_datasets(input_dirs, output_dir, seed=None, workers=8):
    # Optionnel : fixer la seed pour un shuffle reproductible
    if seed is not None:
        random.seed(seed)

    # Crée les dossiers de sortie
    images_out = os.path.join(output_dir, "images")
    points_out = os.path.join(output_dir, "points")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(points_out, exist_ok=True)

    # 1) Récupérer toutes les frames de tous les datasets
    samples = collect_samples(input_dirs)

    if not samples:
        print("Aucun sample trouvé, vérifie les chemins d'entrée.")
        return

    print(f"{len(samples)} frames trouvées au total.")

    # 2) Mélanger (shuffle) les samples
    random.shuffle(samples)

    # 3) Copier avec nouvelle numérotation — en parallèle via thread pool
    tasks = [
        (new_idx, dataset_dir, old_frame_id, images_out, points_out)
        for new_idx, (dataset_dir, old_frame_id) in enumerate(samples)
    ]

    done_count = 0
    total = len(tasks)
    t_start = time.time()
    last_report = t_start
    print(f"Copie en parallèle avec {workers} threads...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_copy_single_frame, t): t for t in tasks}
        for future in as_completed(futures):
            new_idx, success, warnings = future.result()
            for w in warnings:
                print(w)
            done_count += 1
            now = time.time()
            if now - last_report >= 10:
                elapsed = now - t_start
                fps = done_count / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:.0f}s] {done_count}/{total} frames copiées ({fps:.1f} frames/s)")
                last_report = now

    elapsed_total = time.time() - t_start
    print(f"Fusion terminée ✅  ({done_count}/{total} frames copiées en {elapsed_total:.1f}s)")
    print(f"Dossier final : {output_dir}")
    print(f"- Images : {images_out}")
    print(f"- Points : {points_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fusionner plusieurs datasets (images + points) avec renumérotation aléatoire."
    )
    parser.add_argument(
        "--inputs", "-i", nargs="+", required=True,
        help="Liste des dossiers d'entrée (chaque dossier contient 'images/' et 'points/')."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Dossier de sortie."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed pour le shuffle (optionnel, pour rendre le mélange reproductible)."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=24,
        help="Nombre de threads pour la copie parallèle (défaut: 8)."
    )

    args = parser.parse_args()
    merge_datasets(args.inputs, args.output, args.seed, args.workers)
