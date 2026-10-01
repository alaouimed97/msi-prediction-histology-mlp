import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, roc_auc_score

ici = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ici, "figures"), exist_ok=True)

resultats = pd.read_csv(os.path.join(ici, "results_folds.csv"))
predictions = pd.read_csv(os.path.join(ici, "patient_predictions.csv"))

bleu, orange = "#2a78d6", "#eb6834"
texte, texte_secondaire, grille, fond = "#0b0b0b", "#52514e", "#e2e1dd", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": texte,
                     "xtick.color": texte_secondaire, "ytick.color": texte_secondaire})


def style(ax):
    ax.set_facecolor(fond)
    ax.grid(color=grille, linewidth=0.8)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)


def enregistrer(fig, nom):
    fig.savefig(os.path.join(ici, "figures", nom), dpi=150, facecolor=fond, bbox_inches="tight")
    plt.close(fig)


# 1. patient-level ROC curves, one per repetition of the cross-validation
fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor=fond)
for nom, couleur in (("MLP", bleu), ("Logistic regression", orange)):
    aucs = []
    for repetition, groupe in predictions[predictions["model"] == nom].groupby("repetition"):
        fpr, tpr, _ = roc_curve(groupe["msi"], groupe["proba"])
        aucs.append(roc_auc_score(groupe["msi"], groupe["proba"]))
        ax.plot(fpr, tpr, color=couleur, linewidth=2, alpha=0.85,
                label=f"{nom} (AUC {np.mean(aucs):.3f})" if repetition == 3 else None)
ax.plot([0, 1], [0, 1], color=texte_secondaire, linewidth=1, linestyle="--", label="Random (AUC 0.5)")
ax.set_xlabel("1 - specificity (MSS patients flagged as MSI)")
ax.set_ylabel("Sensitivity (MSI patients found)")
ax.set_title("Patient-level ROC curves, 360 patients\n(one curve per repetition of the cross-validation)",
             color=texte, fontsize=11)
ax.legend(frameon=False, loc="lower right")
style(ax)
enregistrer(fig, "roc_curves.png")

# 2. MLP against logistic regression, fold by fold (same test patients)
auc = resultats[resultats["threshold"] == "inner CV"].pivot(index="fold", columns="model", values="auc")
fig, ax = plt.subplots(figsize=(5.4, 5.2), facecolor=fond)
mieux = auc["MLP"] > auc["Logistic regression"]
ax.scatter(auc.loc[mieux, "Logistic regression"], auc.loc[mieux, "MLP"], s=60, color=bleu,
           edgecolor=fond, linewidth=1.5, zorder=3, label=f"MLP better ({mieux.sum()} folds)")
ax.scatter(auc.loc[~mieux, "Logistic regression"], auc.loc[~mieux, "MLP"], s=60, color=orange,
           edgecolor=fond, linewidth=1.5, zorder=3, label=f"Logistic regression better ({(~mieux).sum()} folds)")
bornes = [0.6, 1.0]
ax.plot(bornes, bornes, color=texte_secondaire, linewidth=1, linestyle="--")
ax.set_xlim(bornes)
ax.set_ylim(bornes)
ax.set_xlabel("Logistic regression AUC")
ax.set_ylabel("MLP AUC")
ax.set_title("Patient-level AUC on each of the 15 test folds", color=texte, fontsize=11)
ax.legend(frameon=False, loc="lower right")
style(ax)
enregistrer(fig, "auc_per_fold.png")

# 3. effect of the decision threshold on the MLP
mlp = resultats[resultats["model"] == "MLP"].groupby("threshold", sort=False)[["sensitivity", "specificity"]].mean()
fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=fond)
positions = np.arange(2)
largeur = 0.36
for k, (colonne, couleur, nom) in enumerate((("sensitivity", bleu, "Sensitivity (MSI found)"),
                                             ("specificity", orange, "Specificity (MSS found)"))):
    barres = ax.bar(positions + (k - 0.5) * largeur, mlp[colonne], largeur - 0.03, color=couleur, label=nom)
    for barre, valeur in zip(barres, mlp[colonne]):
        ax.text(barre.get_x() + barre.get_width() / 2, valeur + 0.02, f"{valeur:.2f}",
                ha="center", color=texte, fontsize=10)
ax.set_xticks(positions)
ax.set_xticklabels(["Default threshold 0.5", "Threshold chosen in the inner CV\n(target: 80 % of MSI patients found)"])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Mean over the 15 test folds")
ax.set_title("MLP: effect of the decision threshold", color=texte, fontsize=11)
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
style(ax)
ax.grid(axis="x", visible=False)
enregistrer(fig, "threshold_effect.png")

# paired test on the 15 folds: is the difference between the MLP and the logistic regression significant?
difference = auc["MLP"] - auc["Logistic regression"]
print(f"MLP - logistic regression: {difference.mean():+.3f} AUC | MLP better in {mieux.sum()}/15 folds | "
      f"Wilcoxon p = {wilcoxon(auc['MLP'], auc['Logistic regression']).pvalue:.2f}")
print("Figures saved in figures/")
