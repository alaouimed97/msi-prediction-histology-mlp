import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ici = os.path.dirname(os.path.abspath(__file__))
dossier_images = os.path.join(ici, "data")


class Tuiles(Dataset):
    def __init__(self, fichiers):
        self.fichiers = fichiers
        # the tiles are already 224 x 224 and colour-normalised: only the ImageNet normalisation is applied
        self.preparation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.fichiers)

    def __getitem__(self, i):
        image = Image.open(os.path.join(dossier_images, self.fichiers[i])).convert("RGB")
        return self.preparation(image)


if __name__ == "__main__":
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # list of the tiles with the patient and the MSI status of each one
    lignes = []
    for classe, msi in (("MSIMUT_JPEG", 1), ("MSS_JPEG", 0)):
        for nom in sorted(os.listdir(os.path.join(dossier_images, classe))):
            patient = re.search(r"TCGA-\w{2}-\w{4}", nom).group(0)
            lignes.append({"fichier": f"{classe}/{nom}", "patient": patient, "msi": msi})
    tuiles = pd.DataFrame(lignes)
    if limite:
        tuiles = tuiles.sample(limite, random_state=0).reset_index(drop=True)
    print(len(tuiles), "tiles |", tuiles["patient"].nunique(), "patients")

    # ResNet18 pre-trained on ImageNet, frozen: it only turns each tile into 512 numbers
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    # channels_last makes the convolutions faster on CPU
    resnet = resnet.to(memory_format=torch.channels_last)

    # tiles are processed in chunks saved to disk, so an interrupted run can resume
    suffixe = f"_{limite}" if limite else ""
    dossier_morceaux = os.path.join(ici, f"chunks{suffixe}")
    os.makedirs(dossier_morceaux, exist_ok=True)
    taille_morceau = 8192
    debut = time.time()
    for k, depart in enumerate(range(0, len(tuiles), taille_morceau)):
        chemin = os.path.join(dossier_morceaux, f"{k:03d}.npy")
        if os.path.exists(chemin):
            continue
        fichiers = tuiles["fichier"].tolist()[depart:depart + taille_morceau]
        chargeur = DataLoader(Tuiles(fichiers), batch_size=128, num_workers=2)
        morceau = []
        with torch.inference_mode():
            for lot in chargeur:
                morceau.append(resnet(lot.contiguous(memory_format=torch.channels_last)).numpy())
        np.save(chemin, np.concatenate(morceau))
        fait = min(depart + taille_morceau, len(tuiles))
        print(f"{fait}/{len(tuiles)} tiles | {(time.time() - debut) / 60:.1f} min", flush=True)

    morceaux = sorted(os.listdir(dossier_morceaux))
    features = np.concatenate([np.load(os.path.join(dossier_morceaux, m)) for m in morceaux])
    assert len(features) == len(tuiles)
    np.save(os.path.join(ici, f"features{suffixe}.npy"), features)
    tuiles.to_csv(os.path.join(ici, f"tiles{suffixe}.csv"), index=False)
    print("Saved", f"features{suffixe}.npy", "and", f"tiles{suffixe}.csv", features.shape)
