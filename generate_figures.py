"""
Geração de figuras de alta qualidade para os 3 modelos finais (CatBoost):
M1 (notificação), M2 (pós-notificação), M3 (evolução clínica).

Para cada modelo, gera:
  - confusion_matrix.png       (matriz de confusão normalizada por classe real)
  - probability_histogram.png  (histograma de probabilidades estimadas - acertos/erros)
  - roc_pr_curves.png           (ROC e Precision-Recall lado a lado)
  - calibration_curve.png       (curva de calibração)
  - shap_summary.png            (beeswarm SHAP)
  - shap_importance.png         (importância média absoluta dos atributos)
  - feature_importance.png      (importância nativa do CatBoost)

E figuras consolidadas:
  - figure1_combined.png        (reproduz a Fig. 1 do paper - 3x2 painel)
  - figure2_shap_combined.png   (reproduz a Fig. 2 do paper - 1x3 painel)
  - algorithm_comparison.png    (5 algoritmos x 3 bases, PR-AUC com erro)
  - performance_evolution.png   (evolução M1 -> M2 -> M3)
  - metrics_radar.png           (radar das métricas operacionais)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/claude")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Estilo limpo, alinhado ao paper (escala de cinza com acentos)
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
})

# Paleta sóbria
COLOR_NEG     = "#4F6D7A"   # azul-acinzentado para Sobreviveu
COLOR_POS     = "#C44536"   # vermelho-tijolo para Óbito
COLOR_HIT     = "#3B5F6F"
COLOR_MISS    = "#D7CFC0"
COLOR_NEUTRAL = "#777777"
COLOR_M1      = "#7A9CC6"
COLOR_M2      = "#5A7B9A"
COLOR_M3      = "#264653"

# Injetar wrappers no __main__ (pickle do joblib espera no __main__)
import wrappers
import __main__
for _n in ["FinalModelWithThreshold", "ProbaCalibratedWrapper", "ProbCalibrator",
          "SafeCatBoostClassifier", "NominalPreprocessor", "get_proba"]:
    setattr(__main__, _n, getattr(wrappers, _n))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import shap

OUT = "./images"
os.makedirs(OUT, exist_ok=True)

# Diretórios padrão dentro do repositório
BASES = "./bases"
MODELS = "./models"

# ---------- Configuração das três bases ----------
CONFIG = {
    "M1": {
        "name": "M1 — Notificação",
        "base_file": f"{BASES}/baseModelo1-VigilanciaNaNotificacao.xlsx",
        "model_file": f"{MODELS}/final_model_wrapped_catboost_model1.joblib",
        "drop_cols": [],
    },
    "M2": {
        "name": "M2 — Pós-notificação",
        "base_file": f"{BASES}/baseModelo2-RiscoClinicoInicial.xlsx",
        "model_file": f"{MODELS}/final_model_wrapped_catboost_model2.joblib",
        "drop_cols": [],
    },
    "M3": {
        "name": "M3 — Evolução clínica",
        "base_file": f"{BASES}/baseModelo3-FatoresAssociadosAoObito.xlsx",
        "model_file": f"{MODELS}/final_model_wrapped_catboost_model3.joblib",
        "drop_cols": ["HOSPITALIZACAO"],
    },
}

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------- Helpers ----------
def load_holdout(cfg):
    df = pd.read_excel(cfg["base_file"])
    for c in cfg["drop_cols"]:
        if c in df.columns:
            df = df.drop(columns=c)
    X = df.drop(columns=["OBITO"])
    y = df["OBITO"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return Xtr, Xte, ytr, yte


def get_inner_catboost(wrapped):
    """Desembrulha o wrapper para chegar no CatBoost (para SHAP/feature_importance nativa)."""
    obj = wrapped.model
    while True:
        if hasattr(obj, "base_estimator"):
            obj = obj.base_estimator; continue
        if hasattr(obj, "calibrated_classifiers_"):
            obj = obj.calibrated_classifiers_[0].estimator; continue
        break
    return obj  # Pipeline com 'prep' + 'model'


def transform_for_shap(pipeline, X):
    """Aplica o preprocessor para gerar X no formato consumido pelo CatBoost interno.
    Para garantir coloração no SHAP (que requer dtype numérico), todos os
    atributos são convertidos para float — categóricos passam a ser tratados
    como numéricos ordenados, o que produz cores graduadas no beeswarm."""
    prep = pipeline.named_steps.get("prep")
    Xp = prep.transform(X)
    if not isinstance(Xp, pd.DataFrame):
        Xp = pd.DataFrame(Xp, columns=X.columns, index=X.index)
    # Forçar tudo a numérico para SHAP colorir todas as variáveis
    Xnum = Xp.copy()
    for c in Xnum.columns:
        if Xnum[c].dtype.name == "category":
            Xnum[c] = Xnum[c].cat.codes.astype(float)
        else:
            Xnum[c] = pd.to_numeric(Xnum[c], errors="coerce").astype(float)
    return Xnum


def compute_full_metrics(y, p, t):
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    spec = tn / (tn + fp + 1e-12)
    npv  = tn / (tn + fn + 1e-12)
    return {
        "ROC-AUC":   roc_auc_score(y, p),
        "PR-AUC":    average_precision_score(y, p),
        "Brier":     brier_score_loss(y, p),
        "Sens":      recall_score(y, yhat),
        "Prec":      precision_score(y, yhat, zero_division=0),
        "F1":        f1_score(y, yhat, zero_division=0),
        "BalAcc":    balanced_accuracy_score(y, yhat),
        "Espec":     spec,
        "VPN":       npv,
        "MCC":       matthews_corrcoef(y, yhat),
        "Youden":    recall_score(y, yhat) + spec - 1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": t,
    }


# ---------- Loop principal ----------
results = {}
print("Carregando bases e modelos, computando predições no holdout...")
for tag, cfg in CONFIG.items():
    print(f"\n>>> {tag}: {cfg['name']}")
    Xtr, Xte, ytr, yte = load_holdout(cfg)
    print(f"   Train: {Xtr.shape}, Test: {Xte.shape}")
    print(f"   Positivos no holdout: {int(yte.sum())}/{len(yte)} ({100*yte.mean():.1f}%)")

    wrapped = joblib.load(cfg["model_file"])
    t = wrapped.threshold
    print(f"   threshold (t*): {t:.3f}")

    proba = wrapped.predict_proba(Xte)[:, 1]
    pred = (proba >= t).astype(int)

    metrics = compute_full_metrics(yte, proba, t)
    print(f"   ROC-AUC={metrics['ROC-AUC']:.3f}  PR-AUC={metrics['PR-AUC']:.3f}  "
          f"Brier={metrics['Brier']:.3f}  Sens={metrics['Sens']:.3f}  "
          f"Espec={metrics['Espec']:.3f}")

    results[tag] = {
        "cfg": cfg, "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte,
        "wrapped": wrapped, "proba": proba, "pred": pred, "t": t,
        "metrics": metrics,
    }


# ============================================================
# FIG 1 (combinado, painel 3x2 — reproduz a Figura 1 do paper)
# ============================================================
print("\nGerando figure1_combined.png ...")
fig, axes = plt.subplots(3, 2, figsize=(11, 11),
                         gridspec_kw={"width_ratios": [1.0, 1.4]})
panel_letters = ["a", "b", "c", "d", "e", "f"]
for i, tag in enumerate(["M1", "M2", "M3"]):
    R = results[tag]
    yte = R["yte"]; pred = R["pred"]; proba = R["proba"]; t = R["t"]
    cm = confusion_matrix(yte, pred, normalize="true")
    cm_abs = confusion_matrix(yte, pred)

    # --- esquerda: matriz de confusão normalizada ---
    ax = axes[i, 0]
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sobreviveu", "Óbito"])
    ax.set_yticklabels(["Sobreviveu", "Óbito"])
    ax.set_ylabel("Real")
    if i == 2:
        ax.set_xlabel("Predição")
    ax.set_title(f"({panel_letters[2*i]}) Matriz de Confusão | {tag}")
    ax.grid(False)
    for ii in range(2):
        for jj in range(2):
            value = cm[ii, jj]
            ax.text(jj, ii - 0.08, f"{value*100:.1f}%",
                    ha="center", va="center",
                    color="white" if value > 0.5 else "black",
                    fontweight="bold", fontsize=14)
            ax.text(jj, ii + 0.18, f"({cm_abs[ii, jj]})",
                    ha="center", va="center",
                    color="white" if value > 0.5 else "#444",
                    fontsize=9)

    # --- direita: histograma de probabilidades ---
    ax = axes[i, 1]
    pos_mask = (yte == 1)
    proba_pos = proba[pos_mask]
    hits = proba_pos >= t
    bins = np.linspace(0.0, 1.0, 31)
    ax.hist(proba_pos[hits],  bins=bins, color=COLOR_HIT,
            label=f"Acerto (n = {int(hits.sum())})", alpha=0.95)
    ax.hist(proba_pos[~hits], bins=bins, color=COLOR_MISS,
            label=f"Erro (n = {int((~hits).sum())})", alpha=0.95)
    ax.axvline(t, color="black", linestyle="--", linewidth=1.0,
               label=f"Limiar = {t:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Frequência")
    if i == 2:
        ax.set_xlabel("Probabilidade estimada de óbito")
    ax.set_title(f"({panel_letters[2*i+1]}) Predição da classe Óbito | {tag}")
    # legenda em caixa para evitar sobreposição visual com as barras altas
    ax.legend(loc="upper right", fontsize=8,
              framealpha=0.92, edgecolor="lightgray")

plt.tight_layout()
plt.savefig(f"{OUT}/figure1_combined.png")
plt.close()
print("OK")


# ============================================================
# Figuras individuais por modelo
# ============================================================
for tag in ["M1", "M2", "M3"]:
    R = results[tag]; cfg = R["cfg"]
    yte = R["yte"]; proba = R["proba"]; pred = R["pred"]; t = R["t"]; m = R["metrics"]
    sub = f"{OUT}/{tag}"
    os.makedirs(sub, exist_ok=True)
    print(f"\nFiguras individuais para {tag} -> {sub}")

    # --- 1) Matriz de confusão (estilo paper) ---
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    cm = confusion_matrix(yte, pred, normalize="true")
    cm_abs = confusion_matrix(yte, pred)
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sobreviveu", "Óbito"])
    ax.set_yticklabels(["Sobreviveu", "Óbito"])
    ax.set_xlabel("Predição"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {tag} ({cfg['name']})\n"
                 f"limiar = {t:.3f}", fontsize=11)
    ax.grid(False)
    for ii in range(2):
        for jj in range(2):
            v = cm[ii, jj]
            ax.text(jj, ii - 0.08, f"{v*100:.1f}%",
                    ha="center", va="center",
                    color="white" if v > 0.5 else "black",
                    fontweight="bold", fontsize=14)
            ax.text(jj, ii + 0.18, f"({cm_abs[ii, jj]})",
                    ha="center", va="center",
                    color="white" if v > 0.5 else "#444", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{sub}/confusion_matrix.png"); plt.close()

    # --- 2) Histograma de probabilidades (Acerto x Erro na classe positiva) ---
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    pos = (yte == 1)
    pp = proba[pos]; hits = pp >= t
    bins = np.linspace(0, 1, 31)
    ax.hist(pp[hits],  bins=bins, color=COLOR_HIT,
            label=f"Acerto (n = {int(hits.sum())})")
    ax.hist(pp[~hits], bins=bins, color=COLOR_MISS,
            label=f"Erro (n = {int((~hits).sum())})")
    ax.axvline(t, color="black", linestyle="--", label=f"Limiar = {t:.3f}")
    ax.set_xlabel("Probabilidade estimada de óbito")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Predição da classe Óbito — {tag}")
    ax.set_xlim(0, 1)
    # legenda fora do plot (à direita) para não sobrepor barras
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    plt.savefig(f"{sub}/probability_histogram.png"); plt.close()

    # --- 3) Curvas ROC + PR ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fpr, tpr, _ = roc_curve(yte, proba)
    axes[0].plot(fpr, tpr, color=COLOR_M3, lw=2,
                 label=f"AUC = {m['ROC-AUC']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=COLOR_NEUTRAL, lw=1)
    axes[0].set_xlabel("Taxa de falsos positivos (1 - Especificidade)")
    axes[0].set_ylabel("Taxa de verdadeiros positivos (Sensibilidade)")
    axes[0].set_title(f"ROC — {tag}")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(yte, proba)
    pi = float(yte.mean())
    axes[1].plot(rec, prec, color=COLOR_M3, lw=2,
                 label=f"PR-AUC = {m['PR-AUC']:.3f}")
    axes[1].axhline(pi, color=COLOR_NEUTRAL, linestyle="--", lw=1,
                    label=f"Acaso = {pi:.3f}")
    axes[1].set_xlabel("Sensibilidade (Recall)")
    axes[1].set_ylabel("Precisão")
    axes[1].set_title(f"Precision-Recall — {tag}")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{sub}/roc_pr_curves.png"); plt.close()

    # --- 4) Curva de calibração ---
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=COLOR_M3, lw=2, ms=7,
            label=f"Modelo (Brier = {m['Brier']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=COLOR_NEUTRAL, lw=1, label="Calibração perfeita")
    ax.set_xlabel("Probabilidade média prevista (por bin)")
    ax.set_ylabel("Fração de positivos observada")
    ax.set_title(f"Curva de calibração — {tag}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{sub}/calibration_curve.png"); plt.close()

    # --- 5) Importância nativa do CatBoost ---
    pipeline = get_inner_catboost(R["wrapped"])
    cb = pipeline.named_steps["model"]
    fi = np.asarray(cb.get_feature_importance())
    fnames = list(cb.feature_names_)
    order = np.argsort(fi)
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32*len(fnames)+1)))
    ax.barh(np.array(fnames)[order], fi[order], color=COLOR_M3)
    ax.set_xlabel("Importância (CatBoost — PredictionValuesChange)")
    ax.set_title(f"Importância nativa dos atributos — {tag}")
    plt.tight_layout()
    plt.savefig(f"{sub}/feature_importance.png"); plt.close()


# ============================================================
# SHAP individual e combinado
# ============================================================
print("\nCalculando SHAP para cada modelo...")
shap_data = {}
for tag in ["M1", "M2", "M3"]:
    R = results[tag]
    pipe = get_inner_catboost(R["wrapped"])
    prep = pipe.named_steps["prep"]
    cb   = pipe.named_steps["model"]

    # X com tipos esperados pelo CatBoost (com categóricos quando aplicável)
    Xprep = prep.transform(R["Xte"])
    if not isinstance(Xprep, pd.DataFrame):
        Xprep = pd.DataFrame(Xprep, columns=R["Xte"].columns, index=R["Xte"].index)

    # Versão numérica para a visualização (colorir todas as variáveis)
    Xnum = Xprep.copy()
    for c in Xnum.columns:
        if Xnum[c].dtype.name == "category":
            Xnum[c] = Xnum[c].cat.codes.astype(float)
        else:
            Xnum[c] = pd.to_numeric(Xnum[c], errors="coerce").astype(float)

    explainer = shap.TreeExplainer(cb)
    sv = explainer.shap_values(Xprep)
    if isinstance(sv, list):
        sv = sv[1]
    shap_data[tag] = {"shap": sv, "X": Xnum, "exp": explainer}

    # SHAP summary individual (colorido)
    plt.figure(figsize=(8.5, max(4.0, 0.30*Xnum.shape[1]+1.5)))
    shap.summary_plot(sv, Xnum, show=False, max_display=Xnum.shape[1])
    plt.title(f"SHAP summary — {tag}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}/shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # SHAP importance bar
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32*len(mean_abs)+1)))
    ax.barh(np.array(Xnum.columns)[order], mean_abs[order], color=COLOR_M3)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"Importância média absoluta SHAP — {tag}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}/shap_importance.png"); plt.close()

# --- Figura 2 combinada (3 painéis SHAP, reproduz Fig.2 do paper) ---
print("\nGerando figure2_shap_combined.png ...")
fig = plt.figure(figsize=(15, 6.5))
for j, tag in enumerate(["M1", "M2", "M3"]):
    ax = plt.subplot(1, 3, j+1)
    sv = shap_data[tag]["shap"]; X = shap_data[tag]["X"]
    plt.sca(ax)
    shap.summary_plot(sv, X, show=False, max_display=X.shape[1],
                      plot_size=None, color_bar=(j == 2))
    ax.set_title(f"({chr(ord('a')+j)}) Gráfico de resumo SHAP | {tag}",
                 fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/figure2_shap_combined.png", dpi=200, bbox_inches="tight")
plt.close()


# ============================================================
# COMPARAÇÃO DE ALGORITMOS (5 x 3 bases) usando os CSVs
# ============================================================
print("\nGerando algorithm_comparison.png ...")
ranks = {
    "M1": pd.read_csv("./results/ranking_modelos_modelo1.csv"),
    "M2": pd.read_csv("./results/ranking_modelos_modelo2.csv"),
    "M3": pd.read_csv("./results/ranking_modelos_modelo3.csv"),
}
algos = ["catboost", "xgboost", "random_forest", "adaboost", "decision_tree"]
algo_label = {"catboost": "CatBoost", "xgboost": "XGBoost",
              "random_forest": "Random Forest", "adaboost": "AdaBoost",
              "decision_tree": "Decision Tree"}
bases = ["M1", "M2", "M3"]
base_colors = [COLOR_M1, COLOR_M2, COLOR_M3]

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
# subplot 1: PR-AUC
ax = axes[0]
x = np.arange(len(algos)); width = 0.27
for j, b in enumerate(bases):
    df = ranks[b].set_index("modelo").loc[algos]
    ax.bar(x + (j-1)*width, df["pr_auc_mean"].values, width,
           yerr=df["pr_auc_std"].values, label=b,
           color=base_colors[j], capsize=3, alpha=0.9)
ax.axhline(0.121, color=COLOR_NEUTRAL, linestyle=":",
           label="Acaso (π=0,121)")
ax.set_xticks(x)
ax.set_xticklabels([algo_label[a] for a in algos], rotation=15)
ax.set_ylabel("PR-AUC (média ± std, nCV)")
ax.set_title("Comparação dos algoritmos — PR-AUC")
# Top headroom para a legenda não sobrepor as barras
top1 = max(ranks[b]["pr_auc_mean"].max() + ranks[b]["pr_auc_std"].max() for b in bases)
ax.set_ylim(0, top1 * 1.30)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=4, frameon=False)

# subplot 2: ROC-AUC
ax = axes[1]
for j, b in enumerate(bases):
    df = ranks[b].set_index("modelo").loc[algos]
    ax.bar(x + (j-1)*width, df["roc_auc_mean"].values, width,
           yerr=df["roc_auc_std"].values, label=b,
           color=base_colors[j], capsize=3, alpha=0.9)
ax.axhline(0.5, color=COLOR_NEUTRAL, linestyle=":", label="Acaso (0,5)")
ax.set_xticks(x)
ax.set_xticklabels([algo_label[a] for a in algos], rotation=15)
ax.set_ylabel("ROC-AUC (média ± std, nCV)")
ax.set_title("Comparação dos algoritmos — ROC-AUC")
ax.set_ylim(0.5, 1.0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=4, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/algorithm_comparison.png"); plt.close()


# ============================================================
# EVOLUÇÃO DO DESEMPENHO M1 -> M2 -> M3
# ============================================================
print("\nGerando performance_evolution.png ...")
metric_to_plot = ["ROC-AUC", "PR-AUC", "Brier"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
xs = np.arange(3)
labels = ["M1", "M2", "M3"]
for i, mname in enumerate(metric_to_plot):
    vals = [results[t]["metrics"][mname] for t in labels]
    ax = axes[i]
    ax.plot(xs, vals, "o-", color=COLOR_M3, lw=2.5, ms=10)
    for xi, vi in zip(xs, vals):
        ax.annotate(f"{vi:.3f}", (xi, vi),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_title(mname)
    ax.set_xlabel("Estágio do acompanhamento clínico")
    if mname == "Brier":
        ax.set_ylim(min(vals)*0.85, max(vals)*1.15)
        ax.set_ylabel("Brier score (menor é melhor)")
    else:
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{mname} (holdout)")
plt.suptitle("Evolução do desempenho do CatBoost ao longo dos estágios",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/performance_evolution.png"); plt.close()


# ============================================================
# RADAR de métricas operacionais
# ============================================================
print("\nGerando metrics_radar.png ...")
metrics_radar = ["Sens", "Espec", "Prec", "VPN", "BalAcc", "F1"]
metrics_label = ["Sensibilidade", "Especificidade", "Precisão",
                 "VPN", "Acurácia\nbalanceada", "F1"]
fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw={"projection": "polar"})
N = len(metrics_radar)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for tag, color in zip(["M1", "M2", "M3"], [COLOR_M1, COLOR_M2, COLOR_M3]):
    vals = [results[tag]["metrics"][m] for m in metrics_radar]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, lw=2, label=tag)
    ax.fill(angles, vals, color=color, alpha=0.18)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_label)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim(0, 1.0)
ax.set_title("Comparativo das métricas operacionais (holdout)",
             fontweight="bold", pad=24)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
          ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT}/metrics_radar.png"); plt.close()


# ============================================================
# HEATMAP de métricas (modelo x métrica)
# ============================================================
print("\nGerando metrics_heatmap.png ...")
hm_metrics = ["ROC-AUC", "PR-AUC", "Brier", "Sens", "Espec", "Prec",
              "F1", "BalAcc", "VPN", "MCC", "Youden"]
M = np.zeros((3, len(hm_metrics)))
for i, tag in enumerate(["M1", "M2", "M3"]):
    M[i] = [results[tag]["metrics"][m] for m in hm_metrics]
fig, ax = plt.subplots(figsize=(11, 3.2))
im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(hm_metrics))); ax.set_xticklabels(hm_metrics, rotation=25)
ax.set_yticks(range(3)); ax.set_yticklabels(["M1", "M2", "M3"])
ax.set_title("Métricas no holdout — modelo final CatBoost",
             fontweight="bold")
for i in range(3):
    for j in range(len(hm_metrics)):
        v = M[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v < 0.55 else "black", fontsize=9, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Valor (0-1)")
ax.grid(False)
plt.tight_layout()
plt.savefig(f"{OUT}/metrics_heatmap.png"); plt.close()


# ============================================================
# Salvar tabela final com TODAS as métricas dos 3 modelos
# ============================================================
print("\nSalvando tabela holdout_metrics_full.csv ...")
rows = []
for tag in ["M1", "M2", "M3"]:
    row = {"Modelo": tag}
    row.update(results[tag]["metrics"])
    rows.append(row)
os.makedirs("./results", exist_ok=True)
pd.DataFrame(rows).to_csv("./results/holdout_metrics_full.csv", index=False)

print("\n========================================")
print("Concluído! Figuras geradas em:", OUT)
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        full = os.path.join(root, f)
        size = os.path.getsize(full) / 1024
        print(f"  {full}  ({size:.0f} KB)")
