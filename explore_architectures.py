import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ici = os.path.dirname(os.path.abspath(__file__))
features = np.load(os.path.join(ici, "features.npy"))
tuiles = pd.read_csv(os.path.join(ici, "tiles.csv"))
labels_tuiles = tuiles["msi"].to_numpy()
patients = tuiles.groupby("patient")["msi"].first()
noms_patients, y_patients = patients.index.to_numpy(), patients.to_numpy()
tuiles_du_patient = tuiles.groupby("patient").indices

# each architecture is evaluated on exactly the same 15 folds as the logistic regression
architectures = {
    "1 layer 256, dropout 0.2 (current)": dict(couches=[256], dropout=0.2),
    "1 layer 512, dropout 0.5": dict(couches=[512], dropout=0.5),
    "2 layers 256-64, dropout 0.3": dict(couches=[256, 64], dropout=0.3),
    "2 layers 512-128, dropout 0.5, batchnorm": dict(couches=[512, 128], dropout=0.5, batchnorm=True),
    "3 layers 512-256-64, dropout 0.5": dict(couches=[512, 256, 64], dropout=0.5),
    "1 layer 256, dropout 0.5, input dropout 0.2": dict(couches=[256], dropout=0.5, dropout_entree=0.2),
    "1 layer 256, dropout 0.5, weight decay 1e-2": dict(couches=[256], dropout=0.5, weight_decay=0.01),
    "1 layer 256, dropout 0.2, 300 tiles/patient": dict(couches=[256], dropout=0.2, max_tuiles=300),
    "2 layers 256-64, dropout 0.3, 300 tiles/patient": dict(couches=[256, 64], dropout=0.3, max_tuiles=300),
    "1 layer 256, dropout 0.2, ensemble of 5 seeds": dict(couches=[256], dropout=0.2, ensemble=5),
}


def tuiles_entrainement(liste_patients, graine, max_tuiles):
    rng = np.random.default_rng(graine)
    indices = []
    for p in liste_patients:
        idx = tuiles_du_patient[p]
        if len(idx) > max_tuiles:
            idx = rng.choice(idx, max_tuiles, replace=False)
        indices.append(idx)
    return np.concatenate(indices)


def construire_mlp(couches, dropout, batchnorm=False, dropout_entree=0.0):
    blocs, entree = [], 512
    if dropout_entree:
        blocs.append(nn.Dropout(dropout_entree))
    for n in couches:
        blocs.append(nn.Linear(entree, n))
        if batchnorm:
            blocs.append(nn.BatchNorm1d(n))
        blocs += [nn.ReLU(), nn.Dropout(dropout)]
        entree = n
    blocs.append(nn.Linear(entree, 1))
    return nn.Sequential(*blocs)


def entrainer(X, y, graine, couches, dropout, batchnorm=False, dropout_entree=0.0, weight_decay=0.0001, epochs=15):
    torch.manual_seed(graine)
    modele = construire_mlp(couches, dropout, batchnorm, dropout_entree)
    perte = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y == 0).sum() / (y == 1).sum(), dtype=torch.float32))
    optimiseur = torch.optim.Adam(modele.parameters(), lr=0.001, weight_decay=weight_decay)
    X_t, y_t = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    generateur = torch.Generator().manual_seed(graine)
    modele.train()
    for epoch in range(epochs):
        ordre = torch.randperm(len(X_t), generator=generateur)
        for debut in range(0, len(X_t), 256):
            lot = ordre[debut:debut + 256]
            if len(lot) < 2:
                continue
            optimiseur.zero_grad()
            perte(modele(X_t[lot]).squeeze(1), y_t[lot]).backward()
            optimiseur.step()
    modele.eval()
    return modele


def proba_patients(proba_tuiles, liste_patients, idx_test):
    serie = pd.Series(proba_tuiles, index=idx_test)
    return np.array([serie.loc[tuiles_du_patient[p]].mean() for p in liste_patients])


validation_croisee = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
plis = list(validation_croisee.split(noms_patients, y_patients))
lignes = []
debut = time.time()

for fold, (i_train, i_test) in enumerate(plis):
    patients_train, patients_test = noms_patients[i_train], noms_patients[i_test]
    y_test = y_patients[i_test]
    idx_test = np.concatenate([tuiles_du_patient[p] for p in patients_test])

    for max_tuiles in (100, 300):
        idx_train = tuiles_entrainement(patients_train, fold, max_tuiles)
        normalisation = StandardScaler().fit(features[idx_train])
        regression = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            normalisation.transform(features[idx_train]), labels_tuiles[idx_train])
        proba = regression.predict_proba(normalisation.transform(features[idx_test]))[:, 1]
        lignes.append({"model": f"Logistic regression, {max_tuiles} tiles/patient", "fold": fold + 1,
                       "auc": roc_auc_score(y_test, proba_patients(proba, patients_test, idx_test))})

    for nom, reglages in architectures.items():
        reglages = dict(reglages)
        max_tuiles = reglages.pop("max_tuiles", 100)
        ensemble = reglages.pop("ensemble", 1)
        idx_train = tuiles_entrainement(patients_train, fold, max_tuiles)
        normalisation = StandardScaler().fit(features[idx_train])
        X_train = normalisation.transform(features[idx_train])
        X_test = torch.tensor(normalisation.transform(features[idx_test]), dtype=torch.float32)
        proba = np.zeros(len(idx_test))
        for k in range(ensemble):
            modele = entrainer(X_train, labels_tuiles[idx_train], 100 * fold + k, **reglages)
            with torch.no_grad():
                proba += torch.sigmoid(modele(X_test).squeeze(1)).numpy() / ensemble
        lignes.append({"model": nom, "fold": fold + 1,
                       "auc": roc_auc_score(y_test, proba_patients(proba, patients_test, idx_test))})

    print(f"fold {fold + 1}/15 done | {(time.time() - debut) / 60:.1f} min", flush=True)

resultats = pd.DataFrame(lignes)
lr = resultats[resultats["model"] == "Logistic regression, 100 tiles/patient"].set_index("fold")["auc"]
resume = resultats.groupby("model", sort=False)["auc"].agg(["mean", "std"])
resume["beats LR (folds /15)"] = [
    int((resultats[resultats["model"] == m].set_index("fold")["auc"] > lr).sum()) for m in resume.index]
print("\nPatient-level AUC on the same 15 folds\n")
print(resume.sort_values("mean", ascending=False).round(3).to_string())
resultats.to_csv(os.path.join(ici, "architectures_folds.csv"), index=False)
resume.round(3).to_csv(os.path.join(ici, "architectures_summary.csv"))
print("\nResults saved.")
