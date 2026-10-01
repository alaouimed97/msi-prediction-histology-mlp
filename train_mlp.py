import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, f1_score

ici = os.path.dirname(os.path.abspath(__file__))
rapide = "--rapide" in sys.argv

features = np.load(os.path.join(ici, "features.npy"))
tuiles = pd.read_csv(os.path.join(ici, "tiles.csv"))

patients = tuiles.groupby("patient")["msi"].first()
noms_patients = patients.index.to_numpy()
y_patients = patients.to_numpy()
print(len(tuiles), "tiles |", len(patients), "patients |", y_patients.sum(), "MSI |", (1 - y_patients).sum(), "MSS")

# index of the tiles of each patient
tuiles_du_patient = tuiles.groupby("patient").indices

max_tuiles_par_patient = 100
epochs = 5 if rapide else 15
# the 3 best architectures of explore_architectures.py: the inner cross-validation chooses between them
grille = {
    "2 layers 512-128, dropout 0.5, batchnorm": dict(couches=[512, 128], dropout=0.5, batchnorm=True),
    "1 layer 512, dropout 0.5": dict(couches=[512], dropout=0.5),
    "2 layers 256-64, dropout 0.3": dict(couches=[256, 64], dropout=0.3),
}


def tuiles_entrainement(liste_patients, graine):
    # at most 100 tiles per patient, so that each patient weighs about the same in training
    rng = np.random.default_rng(graine)
    indices = []
    for p in liste_patients:
        idx = tuiles_du_patient[p]
        if len(idx) > max_tuiles_par_patient:
            idx = rng.choice(idx, max_tuiles_par_patient, replace=False)
        indices.append(idx)
    return np.concatenate(indices)


class MLP(nn.Module):
    def __init__(self, couches, dropout, batchnorm=False):
        super().__init__()
        blocs, entree = [], 512
        for n in couches:
            blocs.append(nn.Linear(entree, n))
            if batchnorm:
                blocs.append(nn.BatchNorm1d(n))
            blocs += [nn.ReLU(), nn.Dropout(dropout)]
            entree = n
        blocs.append(nn.Linear(entree, 1))
        self.reseau = nn.Sequential(*blocs)

    def forward(self, x):
        return self.reseau(x).squeeze(1)


def entrainer_mlp(X, y, architecture, graine):
    torch.manual_seed(graine)
    modele = MLP(**grille[architecture])
    # MSS tiles are more numerous: the positive class is weighted to balance the loss
    poids_positif = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32)
    perte = nn.BCEWithLogitsLoss(pos_weight=poids_positif)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=0.001, weight_decay=0.0001)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    generateur = torch.Generator().manual_seed(graine)
    modele.train()
    for epoch in range(epochs):
        ordre = torch.randperm(len(X_t), generator=generateur)
        for debut in range(0, len(X_t), 256):
            lot = ordre[debut:debut + 256]
            # batchnorm needs at least 2 tiles in a batch
            if len(lot) < 2:
                continue
            optimiseur.zero_grad()
            loss = perte(modele(X_t[lot]), y_t[lot])
            loss.backward()
            optimiseur.step()
    modele.eval()
    return modele


def proba_tuiles_mlp(modele, X):
    with torch.no_grad():
        return torch.sigmoid(modele(torch.tensor(X, dtype=torch.float32))).numpy()


def proba_patients(proba_tuiles, liste_patients, indices_tuiles):
    # the probability of a patient is the mean probability of all his tiles
    serie = pd.Series(proba_tuiles, index=indices_tuiles)
    return np.array([serie.loc[tuiles_du_patient[p]].mean() for p in liste_patients])


def entrainer_et_predire(patients_train, patients_test, architecture, graine):
    idx_train = tuiles_entrainement(patients_train, graine)
    idx_test = np.concatenate([tuiles_du_patient[p] for p in patients_test])
    normalisation = StandardScaler().fit(features[idx_train])
    modele = entrainer_mlp(normalisation.transform(features[idx_train]), tuiles["msi"].to_numpy()[idx_train],
                           architecture, graine)
    proba = proba_tuiles_mlp(modele, normalisation.transform(features[idx_test]))
    return proba_patients(proba, patients_test, idx_test)


def predire_regression(patients_train, patients_test, graine):
    # logistic regression on the same tiles and the same features as the MLP
    idx_train = tuiles_entrainement(patients_train, graine)
    idx_test = np.concatenate([tuiles_du_patient[p] for p in patients_test])
    normalisation = StandardScaler().fit(features[idx_train])
    regression = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        normalisation.transform(features[idx_train]), tuiles["msi"].to_numpy()[idx_train])
    proba = regression.predict_proba(normalisation.transform(features[idx_test]))[:, 1]
    return proba_patients(proba, patients_test, idx_test)


def choisir_reglages(patients_train, y_train, graine):
    # inner cross-validation on the training patients only: the test patients are never used to choose
    interne = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=graine).split(patients_train, y_train))
    resultats_internes = {}
    for architecture in grille:
        proba = np.zeros(len(patients_train))
        aucs = []
        for i_tr, i_va in interne:
            proba[i_va] = entrainer_et_predire(patients_train[i_tr], patients_train[i_va], architecture, graine)
            aucs.append(roc_auc_score(y_train[i_va], proba[i_va]))
        resultats_internes[architecture] = (np.mean(aucs), proba)
    meilleur = max(resultats_internes, key=lambda k: resultats_internes[k][0])
    seuil_mlp = choisir_seuil(y_train, resultats_internes[meilleur][1])

    # same inner folds for the logistic regression, to choose its threshold too
    proba_regression = np.zeros(len(patients_train))
    for i_tr, i_va in interne:
        proba_regression[i_va] = predire_regression(patients_train[i_tr], patients_train[i_va], graine)
    seuil_regression = choisir_seuil(y_train, proba_regression)
    return meilleur, seuil_mlp, seuil_regression


def choisir_seuil(y_vrai, proba):
    # highest threshold that still finds at least 80 % of the MSI patients:
    # missing an MSI patient is worse than an extra genetic test for an MSS patient
    for seuil in np.sort(np.unique(proba))[::-1]:
        if recall_score(y_vrai, (proba >= seuil).astype(int)) >= recall_vise:
            return seuil
    return 0.0


def metriques(y_vrai, proba, seuil):
    prediction = (proba >= seuil).astype(int)
    return {
        "auc": roc_auc_score(y_vrai, proba),
        "balanced_accuracy": balanced_accuracy_score(y_vrai, prediction),
        "sensitivity": recall_score(y_vrai, prediction, pos_label=1, zero_division=0),
        "specificity": recall_score(y_vrai, prediction, pos_label=0, zero_division=0),
        "f1": f1_score(y_vrai, prediction, zero_division=0),
    }


recall_vise = 0.8
n_repetitions = 1 if rapide else 3
# new patient folds (the architectures were selected on the folds of random_state=42)
validation_croisee = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repetitions, random_state=2026)
lignes, choix, predictions = [], [], []
debut = time.time()

for fold, (i_train, i_test) in enumerate(validation_croisee.split(noms_patients, y_patients)):
    patients_train, patients_test = noms_patients[i_train], noms_patients[i_test]
    y_train, y_test = y_patients[i_train], y_patients[i_test]
    repetition = fold // 5

    architecture, seuil_mlp, seuil_regression = choisir_reglages(patients_train, y_train, graine=fold)
    choix.append({"fold": fold + 1, "architecture": architecture,
                  "threshold_mlp": round(seuil_mlp, 3), "threshold_lr": round(seuil_regression, 3)})

    proba_mlp = entrainer_et_predire(patients_train, patients_test, architecture, graine=fold)
    proba_regression = predire_regression(patients_train, patients_test, graine=fold)
    proba_majorite = np.full(len(patients_test), float(y_train.mean() >= 0.5))

    for nom, proba, seuil in (("MLP", proba_mlp, seuil_mlp),
                              ("Logistic regression", proba_regression, seuil_regression),
                              ("Majority class", proba_majorite, 0.5)):
        # the same predictions evaluated with the default threshold 0.5 and with the threshold of the inner CV
        lignes.append({"model": nom, "threshold": "0.5", "fold": fold + 1, **metriques(y_test, proba, 0.5)})
        lignes.append({"model": nom, "threshold": "inner CV", "fold": fold + 1, **metriques(y_test, proba, seuil)})
        predictions += [{"model": nom, "repetition": repetition + 1, "patient": p, "msi": v, "proba": q, "threshold": seuil}
                        for p, v, q in zip(patients_test, y_test, proba)]

    print(f"fold {fold + 1}/{5 * n_repetitions} | {architecture} | "
          f"AUC MLP={lignes[-5]['auc']:.3f} LR={lignes[-3]['auc']:.3f} | "
          f"recall MLP {lignes[-6]['sensitivity']:.2f} -> {lignes[-5]['sensitivity']:.2f} (threshold {seuil_mlp:.2f}) | "
          f"{(time.time() - debut) / 60:.1f} min", flush=True)

resultats = pd.DataFrame(lignes)
resume = resultats.groupby(["model", "threshold"], sort=False).agg(["mean", "std"]).drop(columns="fold")
print("\nPatient-level results (mean +/- std over the folds)\n")
print(resume.round(3).to_string())
print("\nSettings chosen by the inner cross-validation:")
print(pd.DataFrame(choix)["architecture"].value_counts().to_string())
print(pd.DataFrame(choix)[["threshold_mlp", "threshold_lr"]].describe().round(3).to_string())

suffixe = "_rapide" if rapide else ""
resultats.to_csv(os.path.join(ici, f"results_folds{suffixe}.csv"), index=False)
resume.round(3).to_csv(os.path.join(ici, f"results_summary{suffixe}.csv"))
pd.DataFrame(choix).to_csv(os.path.join(ici, f"chosen_settings{suffixe}.csv"), index=False)
pd.DataFrame(predictions).to_csv(os.path.join(ici, f"patient_predictions{suffixe}.csv"), index=False)
print("\nResults saved.")
