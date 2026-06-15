"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         GESICA — ANALYSE COVID-19 FRANCHE-COMTÉ                             ║
║         Nassir Ousmane — M1 Ingénierie de la Santé                          ║
║         Laboratoire SINeRGIE — Projet Gesica / Interreg                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
LANCER : streamlit run dashboard_gesica_v2.py
FICHIERS REQUIS : resultats_sirs_.csv
"""

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import correlate
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Gesica — Urgences COVID-19 FC",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg,#0d1b2a 0%,#112240 60%,#0d1b2a 100%); }
    .main-header {
        background: linear-gradient(90deg,#0a4f76,#1a7fa8,#00c9a7);
        padding: .9rem 1.6rem; border-radius: 12px; margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,201,167,.25);
    }
    .main-header h1 { color:#fff!important; font-size:1.45rem!important; font-weight:800!important; margin:0!important; }
    .main-header p  { color:rgba(255,255,255,.8)!important; margin:.2rem 0 0!important; font-size:.82rem!important; }
    .kpi-card {
        background: linear-gradient(135deg,#112240,#1a3a5c);
        border:1px solid rgba(0,201,167,.25); border-radius:10px;
        padding:.8rem .6rem; text-align:center;
        box-shadow:0 2px 10px rgba(0,0,0,.3);
    }
    .kpi-val   { font-size:1.5rem; font-weight:800; display:block; }
    .kpi-lbl   { font-size:.65rem; color:#8892b0; text-transform:uppercase; margin-top:.25rem; }
    .kpi-sub   { font-size:.65rem; color:rgba(255,255,255,.55); margin-top:.15rem; }
    .sec-title {
        font-size:.95rem; font-weight:700; color:#00c9a7;
        border-bottom:2px solid rgba(0,201,167,.3);
        padding-bottom:.4rem; margin-bottom:.9rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0a2540,#0d1b2a);
        border-right:1px solid rgba(0,201,167,.18);
    }
    .stTabs [data-baseweb="tab-list"] {
        background:rgba(17,34,64,.85); border-radius:9px; padding:3px; gap:3px;
    }
    .stTabs [data-baseweb="tab"]      { color:#8892b0!important; border-radius:6px; font-size:.78rem; }
    .stTabs [aria-selected="true"]    { background:rgba(0,201,167,.18)!important; color:#00c9a7!important; }
    .footer {
        background:rgba(10,79,118,.18); border-top:1px solid rgba(0,201,167,.18);
        padding:.7rem 1rem; border-radius:8px; margin-top:1.2rem;
        text-align:center; color:#8892b0; font-size:.68rem;
    }
    .logo-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background: rgba(10,79,118,0.2);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES SIRS
# =============================================================================
ALPHA  = 0.4502;  BETA0 = 2.0288;  BETA1 = -0.4731
GAMMA  = 0.3830;  DELTA = 0.0121
MU     = 7.3189;  SIGMA = 1.4358
R0     = BETA0 / GAMMA
POPULATION_FC = 1_176_000

R2_TRAIN = 0.5906;  R2_TEST = 0.7018
RMSE_VAL = 1956;    MAE_VAL = 1189

BOOTSTRAP = {
    "α": {"moy":0.3979, "std":0.0760, "ci":(0.262,0.536)},
    "β₀": {"moy":2.14,   "std":0.19,   "ci":(1.761,2.525)},
    "γ": {"moy":0.351,  "std":0.031,  "ci":(0.290,0.413)},
    "δ": {"moy":0.0143, "std":0.0029, "ci":(0.0085,0.0202)},
    "R₀": {"moy":6.27,   "std":0.72,   "ci":(5.02,7.84)},
}

# Seuils log-normale (statiques)
S75 = stats.lognorm.ppf(0.75, s=SIGMA, scale=np.exp(MU))
S90 = stats.lognorm.ppf(0.90, s=SIGMA, scale=np.exp(MU))
S95 = stats.lognorm.ppf(0.95, s=SIGMA, scale=np.exp(MU))
S99 = stats.lognorm.ppf(0.99, s=SIGMA, scale=np.exp(MU))

# Palette matplotlib
plt.rcParams.update({
    'figure.facecolor': '#112240', 'axes.facecolor': '#0d1b2a',
    'axes.edgecolor': '#334155', 'axes.labelcolor': '#ccd6f6',
    'xtick.color': '#8892b0', 'ytick.color': '#8892b0',
    'text.color': '#ccd6f6', 'grid.color': '#1e3a5f', 'grid.alpha': .4,
})
C = ['#00c9a7', '#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#a8dadc']

# =============================================================================
# HELPERS
# =============================================================================
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df, None
    except FileNotFoundError:
        return None, f"Fichier introuvable : **{path}**"
    except Exception as e:
        return None, str(e)

def find_col(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

def dark_fig(nrows=1, ncols=1, figsize=(12, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('#112240')
    if nrows == 1 and ncols == 1:
        axes.set_facecolor('#0d1b2a')
        return fig, axes
    for ax in np.array(axes).flat:
        ax.set_facecolor('#0d1b2a')
    return fig, axes

def niveau_alerte(val):
    if val > S99:
        return "⚫ Crise", "#888"
    if val > S95:
        return "🔴 Alerte", "#e63946"
    if val > S90:
        return "🟠 Pré-alerte", "#f4a261"
    if val > S75:
        return "🟢 Vigilance", "#2a9d8f"
    return " Normal", "#00c9a7"

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
with st.spinner(" Chargement des données SIRS..."):
    df_res, err_res = load_csv("resultats_sirs_.csv")

with st.spinner(" Chargement des données COVID brutes..."):
    df_raw, err_raw = load_csv("covid-19-passages-aux-urgences-et-actes-sos-medecins-departement-2.csv")
COL_I = find_col(df_res, ['I_taux_estime_infectes', 'I_estime', 'I_taux'])
COL_OBS = find_col(df_res, ['taux_observe', 'taux_obs', 'taux'])
COL_SIM = find_col(df_res, ['taux_simule', 'taux_sim'])
COL_ICLO = find_col(df_res, ['I_ci_lower', 'ci_lower'])
COL_ICHI = find_col(df_res, ['I_ci_upper', 'ci_upper'])

if df_res is not None and COL_I:
    I_all = df_res[COL_I].values
    I_cur = float(I_all[-1])
    I_max = float(I_all.max())
    idx_pic = int(np.argmax(I_all))
    pic_date = df_res['date'].iloc[idx_pic].strftime('%b %Y') if 'date' in df_res.columns else "N/A"
    last_date = df_res['date'].iloc[-1].strftime('%d/%m/%Y') if 'date' in df_res.columns else "N/A"
else:
    I_all = np.array([])
    I_cur = I_max = 0
    pic_date = last_date = "N/A"

niv_txt, niv_col = niveau_alerte(I_cur)

# =============================================================================
# EN-TÊTE AVEC DEUX IMAGES CÔTE À CÔTE
# =============================================================================
def load_img(p):
    try:
        return Image.open(p) if os.path.exists(p) else None
    except:
        return None

# Chargement des images
logo_gesica = load_img("images/logo_gesica.png")
logo_interreg = load_img("images/logo_interreg.png")
image_covid = load_img("images/gesica.png")  # Image COVID
image_projet = load_img("images/projet_gesica.png")  # Nouvelle image (à mettre)

# ================================================================
# LIGNE 1 : DEUX IMAGES CÔTE À CÔTE
# ================================================================
from PIL import Image

def resize_image(img, target_height):
    """Redimensionne une image en gardant le ratio"""
    ratio = target_height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, target_height))

col_img1, col_img2 = st.columns(2)

with col_img1:
    if image_covid:
        # Redimensionner avant d'afficher
        img_resized = resize_image(image_covid, 200)
        st.image(img_resized, use_container_width=False, width=img_resized.width)
    else:
        st.markdown("""...""", unsafe_allow_html=True)

with col_img2:
    if image_projet:
        # Redimensionner avant d'afficher
        img_resized = resize_image(image_projet, 200)
        st.image(img_resized, use_container_width=False, width=img_resized.width)
    else:
        st.markdown("""...""", unsafe_allow_html=True)
# ================================================================
# LIGNE 2 : TITRE PRINCIPAL
# ================================================================
st.markdown("""
<div class="main-header">
    <h1>  Anticiper la saturation, éclairer la décision</h1>
    <p>Franche-Comté · 2020–2025 · Modèle SIRS · Laboratoire SINeRGIE Besançon · Projet Interreg</p>
</div>
""", unsafe_allow_html=True)
# ================================================================
# LIGNE 3 : LOGOS GESICA + INTERREG (AGRANDIS)
# ================================================================
col_logo1, col_logo2, col_logo3 = st.columns([1, 1, 2])

with col_logo1:
    if logo_gesica:
        st.image(logo_gesica, width=200)  # ← Augmente la taille ici (120, 150, 200)
    else:
        st.markdown(" Gesica")

with col_logo2:
    if logo_interreg:
        st.image(logo_interreg, width=200)  # ← Augmente la taille ici
    else:
        st.markdown(" Interreg")

with col_logo3:
    st.markdown("""
    <div style="background: rgba(10,79,118,0.3); border-radius: 8px; padding: 0.3rem 1rem;">
        <small style="color:#8892b0;">Projet financé par l'Union Européenne — Interreg France-Suisse</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("##  Paramètres dynamiques")
    st.markdown("---")

    capacite = st.slider(" Capacité (lits / 100k)", 10000, 100000, 50000, 5000)
    facteur = st.slider(" Facteur crise (×)", 0.5, 10.0, 1.0, 0.5)




    # ============================================
    # AJOUTER ICI (après facteur, avant le ---)
    # ============================================
    st.markdown("---")
    st.markdown("###  Filtre temporel")
    
    if df_res is not None and 'date' in df_res.columns:
        date_min = df_res['date'].min().date()
        date_max = df_res['date'].max().date()
        date_range = st.date_input(
            "Période d'affichage",
            value=[date_min, date_max],
            min_value=date_min,
            max_value=date_max
            )
        if len(date_range) == 2:
            st.session_state['date_debut'] = pd.to_datetime(date_range[0])
            st.session_state['date_fin'] = pd.to_datetime(date_range[1])












    st.markdown("---")
    st.markdown("###  Paramètres SIRS")
    st.dataframe(
        pd.DataFrame({
            "Paramètre": ["α", "β₀", "β₁", "γ", "δ", "R₀"],
            "Valeur": [f"{ALPHA:.4f}", f"{BETA0:.4f}", f"{BETA1:.4f}",
                       f"{GAMMA:.4f}", f"{DELTA:.5f}", f"{R0:.2f}"],
            "Interprétation": [f"{ALPHA*100:.0f}% recours", "transmission", "saisonnalité",
                               f"{1/GAMMA:.1f} sem.", f"{1/DELTA:.0f} sem.", "reproduction"]
        }), hide_index=True, use_container_width=True
    )
    
    
    
    
    # ============================================
    # AJOUTER ICI (après le tableau des paramètres)
    # ============================================
    with st.expander(" Que signifient ces paramètres ?"):
        st.markdown("""
        | Paramètre | Signification | Valeur |
        |-----------|---------------|--------|
        | **α** | Taux de couplage (infectés comptabilisés) | 45.0% |
        | **β₀** | Taux de transmission moyen | 2.03 /semaine |
        | **β₁** | Amplitude de la saisonnalité | -0.47 |
        | **γ** | Taux de guérison → durée maladie | 2.6 sem. |
        | **δ** | Taux de perte d'immunité → durée immunité | 83 sem. |
        | **R₀** | Taux de reproduction | 5.30 |
        """)
        
        
        
        
        
    st.markdown("---")
    st.markdown("###  Validation")
    st.metric("R² Train", f"{R2_TRAIN:.4f}")
    st.metric("R² Test", f"{R2_TEST:.4f}", delta=f"+{R2_TEST-R2_TRAIN:.4f}")
    st.metric("RMSE", f"{RMSE_VAL:,}")
    st.metric("MAE", f"{MAE_VAL:,}")

    st.markdown("---")
    st.markdown("Nassir Ousmane — M1 Ing. Santé")
    st.markdown("Labo SINeRGIE · Besançon · 2025")

# =============================================================================
# FONCTIONS DYNAMIQUES
# =============================================================================
def risque_dynamique(capacite, facteur=1.0):
    I_scale = np.exp(MU) * facteur
    return 1 - stats.lognorm.cdf(capacite, s=SIGMA, scale=I_scale)

# =============================================================================
# ONGLETS (sans Météo)
# =============================================================================
tabs = st.tabs([
    " I(t) & Modèle",
    " Modèle SIR", 
    " Validation",
    " Distribution",
    " VaR & Monte Carlo",
    " Stress Test",
    " Seuils d'alerte",
    " What if",
    " Bootstrap",
    " Données brutes",
    " À propos"
])

# =============================================================================
# TAB 1 : I(t) & Modèle
# =============================================================================
with tabs[0]:
    st.markdown('<div class="sec-title"> I(t) — Taux réel d\'infectés estimé</div>', unsafe_allow_html=True)
    st.info(f"Relation fondamentale : `I(t) = Taux_observé / α` avec α = {ALPHA:.4f} → "
            f"Le taux réel d'infectés est {1/ALPHA:.2f}× plus élevé que les passages observés.\n\n"
            f" I(t) est une variable latente — reconstruction par modèle, pas un comptage exact.")

    if df_res is not None and COL_I and COL_OBS and COL_SIM:
        st.markdown(f" Dernière semaine disponible : {last_date}")
        
        if len(df_res) > 0:
            last_row = df_res.iloc[-1]
            last_obs = float(last_row[COL_OBS]) if COL_OBS is not None else 0
            last_sim = float(last_row[COL_SIM]) if COL_SIM is not None else 0
            last_I = float(last_row[COL_I]) if COL_I is not None else 0
        else:
            last_obs = last_sim = last_I = 0
        
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(" Taux observé", f"{last_obs:.0f}", "/ 100k")
        d2.metric(" Taux simulé", f"{last_sim:.0f}", "/ 100k")
        d3.metric(" I(t) estimé", f"{last_I:.0f}", "/ 100k")
       
        
       
        
       
       
        # ============================================
        # REMPLACER PAR
        # ============================================
        st.markdown("### Situation actuelle")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style="background: rgba(17,34,64,0.8); border-radius: 10px; padding: 0.8rem; text-align: center;">
                <span style="font-size: 0.8rem; color: #8892b0;">NIVEAU D'ALERTE</span><br>
                <span style="font-size: 1.6rem; font-weight: 800; color: {niv_col};">{niv_txt}</span>
        </div>
        """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: rgba(17,34,64,0.8); border-radius: 10px; padding: 0.8rem; text-align: center;">
                <span style="font-size: 0.8rem; color: #8892b0;">I(t) ACTUEL</span><br>
                <span style="font-size: 1.6rem; font-weight: 800; color: #00c9a7;">{I_cur:.0f}</span>
                <span style="font-size: 0.8rem; color: #8892b0;"> / 100k hab</span>
            </div>
            """, unsafe_allow_html=True)
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("Calibration : Observé vs Simulé")
            fig, ax = dark_fig(figsize=(7, 4))
            ax.plot(df_res['date'], df_res[COL_OBS], 'o', color=C[0], ms=2, alpha=0.6, label='Observé')
            ax.plot(df_res['date'], df_res[COL_SIM], '-', color=C[1], lw=2, label=f'Simulé SIRS (R²={R2_TEST:.3f})')
            ax.set_ylabel("Taux / 100k hab")
            ax.set_title("Calibration SIRS", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("I(t) — Infectés latents avec IC 95%")
            fig, ax = dark_fig(figsize=(7, 4))
            ax.plot(df_res['date'], df_res[COL_I], '-', color='#f4a261', lw=2, label='I(t) estimé')
            if COL_ICLO and COL_ICHI:
                ax.fill_between(df_res['date'], df_res[COL_ICLO], df_res[COL_ICHI],
                                alpha=0.25, color='#f4a261', label='IC 95%')
            for seuil, lbl, clr in [(S75, 'Q75', '#2a9d8f'), (S95, 'Q95', '#f4a261'), (S99, 'Q99', '#e63946')]:
                ax.axhline(seuil, color=clr, ls='--', lw=1, alpha=0.7, label=lbl)
            ax.set_ylabel("I(t) / 100k hab")
            ax.set_title("I(t) avec seuils d'alerte", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if all(c in df_res.columns for c in ['S_sim', 'R_sim']):
            st.markdown(" Dynamique SIRS — Compartiments S, I, R")
            fig, ax = dark_fig(figsize=(12, 4))
            ax.plot(df_res['date'], df_res['S_sim'], '-', color='#2a9d8f', lw=1.5, label='S — Susceptibles')
            ax.plot(df_res['date'], df_res[COL_I], '-', color='#e63946', lw=2, label='I — Infectés estimés')
            ax.plot(df_res['date'], df_res['R_sim'], '-', color='#457b9d', lw=1.5, label='R — Rétablis')
            ax.set_ylabel("Pour 100 000 hab")
            ax.set_title("Évolution des compartiments S, I, R", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        csv_export = df_res.to_csv(index=False)
        st.download_button(" Exporter résultats SIRS (.csv)", csv_export,
                           "resultats_sirs_export.csv", "text/csv")
    else:
        st.warning(" Chargez `resultats_sirs_.csv` pour afficher les graphiques.")










# =============================================================================
# TAB SIR : MODÈLE SIR (COMPARAISON)
# =============================================================================
with tabs[1]:
    st.markdown('<div class="sec-title"> Modèle SIR — Première approche</div>', unsafe_allow_html=True)
    
    st.info("""
    Avant de développer le modèle SIRS, nous avons d'abord calibré un modèle SIR classique.
    Voici les résultats obtenus et les limites qui nous ont conduits à utiliser le SIRS.
    """)
    
    # Chargement des résultats SIR
    df_sir, err_sir = load_csv("resultats_sir.csv")
    
    # Paramètres SIR (issus de ta calibration)
    alpha_sir = 0.4567
    beta_sir = 2.2132
    gamma_sir = 0.4728
    r0_sir = beta_sir / gamma_sir
    r2_sir = 0.4525
    rmse_sir = 1659
    
    # Affichage des métriques SIR
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("α (couplage)", f"{alpha_sir:.4f}", help="Taux de recours")
    col2.metric("β (transmission)", f"{beta_sir:.4f}", help="Taux de transmission")
    col3.metric("γ (guérison)", f"{gamma_sir:.4f}", help=f"Durée = {1/gamma_sir:.1f} sem.")
    col4.metric("R₀", f"{r0_sir:.2f}", help="Taux de reproduction")
    
    st.markdown("---")
    
    # Métriques de performance
    col1, col2 = st.columns(2)
    col1.metric("R² (performance)", f"{r2_sir:.4f}", delta="0.4525")
    col2.metric("RMSE", f"{rmse_sir:.0f}")
    
    # ============================================
    # GRAPHIQUE DE CALIBRATION SIR (AVEC DONNÉES RÉELLES)
    # ============================================
    st.markdown("Calibration du modèle SIR (données réelles)")
    
    if df_sir is not None and 'taux_simule_sir' in df_sir.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_sir['date'], df_sir['taux_observe'], 'bo-', markersize=2, alpha=0.5, label='Observé')
        ax.plot(df_sir['date'], df_sir['taux_simule_sir'], 'r-', linewidth=2, label=f'Simulé SIR (R²={r2_sir:.3f})')
        ax.set_ylabel("Taux / 100k hab")
        ax.set_xlabel("Date")
        ax.set_title("Calibration du modèle SIR")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.caption(" Le modèle SIR reproduit correctement la première vague (printemps 2020) mais ses performances se dégradent pour les vagues suivantes.")
    else:
        st.warning(f" Fichier resultats_sir.csv non trouvé. Exécute d'abord le script SIR. Erreur : {err_sir}")
    
    # ============================================
    # POURQUOI PASSER AU SIRS ?
    # ============================================
    st.markdown("---")
    st.markdown("###  Pourquoi passer du SIR au SIRS ?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
         Limites du modèle SIR :
        - Suppose une immunité définitive (les rétablis ne retombent jamais malades)
        - Ne peut pas reproduire plusieurs vagues successives
        - R² = 0,45 → performance limitée sur toute la période
        
        Données observées :
        - Plusieurs vagues : printemps 2020, automne 2020, vague Omicron (2021-2022)
        - Réinfections possibles (immunité non définitive)
        """)
    
    with col2:
        st.markdown("""
         Pourquoi le SIRS est meilleur :
        - Ajoute un paramètre δ (taux de perte d'immunité)
        - Permet de modéliser des vagues successives
        - R² = 0,70 (test) → +55% de performance
        
        Données du SIRS :
        - Durée d'immunité : 83 semaines (~19 mois)
        - R² test : 0,70 → 70% de variance expliquée
        """)
    
    # Tableau comparatif
    st.markdown("---")
    st.markdown("###  Comparaison SIR vs SIRS")
    
    df_compare = pd.DataFrame({
        "Critère": ["Immunité", "Vagues multiples", "R² (test)", "RMSE", "Paramètres"],
        "Modèle SIR": ["Définitive", " Non", "0,45", "1 659", "3 (α, β, γ)"],
        "Modèle SIRS": ["Temporaire (19 mois)", " Oui", "0,70", "1 956", "5 (α, β₀, β₁, γ, δ)"]
    })
    st.dataframe(df_compare, hide_index=True, use_container_width=True)
    
    # Conclusion
    st.success("""
    Conclusion : Le modèle SIR est utile comme point de départ, mais le modèle SIRS est plus adapté 
    pour représenter la dynamique réelle de l'épidémie sur une période longue (2020-2025) avec plusieurs vagues successives.
    C'est pourquoi nous utilisons le SIRS pour la suite des analyses.
    """)







# =============================================================================
# TAB 2 : Validation
# =============================================================================
with tabs[2]:
    st.markdown('<div class="sec-title"> Validation du modèle</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(" R² Train", f"{R2_TRAIN:.4f}")
    c2.metric(" R² Test", f"{R2_TEST:.4f}", delta=f"+{R2_TEST - R2_TRAIN:.4f} vs train")
    c3.metric(" RMSE", f"{RMSE_VAL:,}", help="Racine de l'erreur quadratique moyenne")
    c4.metric(" MAE", f"{MAE_VAL:,}", help="Erreur absolue moyenne")

    st.success(f" R² test ({R2_TEST:.4f}) > R² train ({R2_TRAIN:.4f}) — Pas de sur-apprentissage. "
               f"Le modèle explique {R2_TEST*100:.1f}% de la variance sur données inédites.")

    if df_res is not None and COL_OBS and COL_SIM:
        residus = (df_res[COL_OBS] - df_res[COL_SIM]).dropna()
        sigma_r = residus.std()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Résidus dans le temps")
            fig, ax = dark_fig(figsize=(7, 4))
            ax.plot(df_res['date'], residus, color='#a8dadc', lw=1, alpha=0.8)
            ax.axhline(0, color='white', ls='--', lw=1.5)
            ax.axhline(2 * sigma_r, color='#f4a261', ls=':', lw=1, alpha=0.7, label='+2σ')
            ax.axhline(-2 * sigma_r, color='#f4a261', ls=':', lw=1, alpha=0.7, label='-2σ')
            ax.fill_between(df_res['date'], 0, residus, where=(residus > 0), alpha=0.2, color='#e63946', label='Sous-estimation')
            ax.fill_between(df_res['date'], 0, residus, where=(residus < 0), alpha=0.2, color='#2a9d8f', label='Sur-estimation')
            ax.set_ylabel("Résidus")
            ax.set_title("Résidus temporels", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown(" Distribution des résidus")
            fig, ax = dark_fig(figsize=(7, 4))
            ax.hist(residus, bins=35, density=True, alpha=0.55, color='#457b9d', edgecolor='#334155')
            x_n = np.linspace(residus.min(), residus.max(), 200)
            ax.plot(x_n, stats.norm.pdf(x_n, 0, sigma_r), '-', color='#00c9a7', lw=2, label='Normale théorique')
            ax.set_xlabel("Résidus")
            ax.set_ylabel("Densité")
            ax.set_title("Distribution des résidus", fontsize=10)
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        sw_p = stats.shapiro(residus[:5000])[1]
        ac = correlate(residus, residus, mode='full')
        lag1 = ac[len(ac) // 2 + 1] / ac[len(ac) // 2]

        st.markdown(" Tests statistiques des résidus")
        t1, t2 = st.columns(2)
        with t1:
            status = " Normaux" if sw_p > 0.05 else " Non normaux (attendu — données épidémiques)"
            st.info(f"Shapiro-Wilk : p = {sw_p:.4f} → {status}")
        with t2:
            status2 = " Faible" if abs(lag1) < 0.3 else " Élevée (attendu — série temporelle)"
            st.info(f"Autocorrélation lag 1 : {lag1:.4f} → {status2}")
    else:
        st.info("ℹ️ Colonnes 'taux_observe' / 'taux_simule' non trouvées.")

# =============================================================================
# TAB 3 : Distribution Log-Normale
# =============================================================================
with tabs[3]:
    st.markdown('<div class="sec-title"> Distribution de I(t) — Loi Log-Normale</div>', unsafe_allow_html=True)

    st.markdown(f"""
    | Loi testée | p-value KS | Verdict |
    |---|---|---|
    | Normale | 0.000 |  Rejetée |
    | Log-Normale | 0.605 |  Acceptée |
    | Gamma | 0.000 |  Rejetée |
    | Binomiale Négative | 0.000 |  Rejetée |

    Paramètres estimés : μ = {MU:.4f} · σ = {SIGMA:.4f}  
    Formule : I(t) ~ LogNormale(μ={MU:.4f}, σ={SIGMA:.4f})
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(" Ajustement Log-Normale")
        fig, ax = dark_fig(figsize=(7, 4))
        if df_res is not None and COL_I:
            I_pos = I_all[I_all > 0]
            ax.hist(I_pos, bins=45, density=True, alpha=0.5, color='#457b9d', edgecolor='#334155', label='Données')
        x_r = np.linspace(1, 80000, 500)
        ax.plot(x_r, stats.lognorm.pdf(x_r, s=SIGMA, scale=np.exp(MU)),
                '-', color='#00c9a7', lw=2.5, label='LogNormale ajustée')
        ax.set_xlabel("I(t) / 100k hab")
        ax.set_ylabel("Densité")
        ax.set_title("Histogramme vs Log-Normale", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(" log I(t) ~ Normale")
        fig, ax = dark_fig(figsize=(7, 4))
        if df_res is not None and COL_I:
            log_I = np.log(I_pos)
            ax.hist(log_I, bins=35, density=True, alpha=0.5, color='#457b9d', edgecolor='#334155', label='log I(t)')
            x_l = np.linspace(log_I.min(), log_I.max(), 200)
        else:
            x_l = np.linspace(3, 12, 200)
        ax.plot(x_l, stats.norm.pdf(x_l, MU, SIGMA), '-', color='#f4a261', lw=2.5,
                label=f'N(μ={MU:.2f}, σ={SIGMA:.2f})')
        ax.set_xlabel("log I(t)")
        ax.set_ylabel("Densité")
        ax.set_title("log I(t) suit une loi Normale", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown(" Tableau des quantiles opérationnels")
    quantiles = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    q_df = pd.DataFrame({
        "Quantile": [f"Q{q*100:.1f}%" for q in quantiles],
        "Seuil I(t)": [f"{stats.lognorm.ppf(q, s=SIGMA, scale=np.exp(MU)):.0f}" for q in quantiles],
        "P(dépasser)": [f"{(1-q)*100:.1f}%" for q in quantiles],
        "Usage": ["Médiane", "Surveillance", "Pré-alerte", "Alerte", "Crise", "Catastrophe"]
    })
    st.dataframe(q_df, hide_index=True, use_container_width=True)

    with st.expander(" Formules pour le rapport"):
        st.latex(r"I(t) \sim \mathcal{LN}(\mu,\sigma), \quad \mu=7.3189,\ \sigma=1.4358")
        st.latex(r"f(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right)")
        st.markdown(f"- **Moyenne théorique :** {np.exp(MU + SIGMA**2/2):.0f}")
        st.markdown(f"- **Intervalle de confiance 95% :** [{np.exp(MU - 1.96*SIGMA):.0f} — {np.exp(MU + 1.96*SIGMA):.0f}]")

# =============================================================================
# TAB 4 : VaR & Monte Carlo
# =============================================================================
with tabs[4]:
    st.markdown('<div class="sec-title"> Value at Risk Sanitaire & Monte Carlo</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("Value at Risk (VaR) sanitaire")
        confs = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
        var_df = pd.DataFrame({
            "Confiance": [f"{c*100:.1f}%" for c in confs],
            "VaR I(t)": [f"{stats.lognorm.ppf(c, s=SIGMA, scale=np.exp(MU) * facteur):.0f}" for c in confs],
            "Risque résiduel": [f"{(1-c)*100:.1f}%" for c in confs],
            "Usage": ["Courant", "Standard", "Pré-alerte", "Alerte", "Crise", "Catastrophe"]
        })
        st.dataframe(var_df, hide_index=True, use_container_width=True)

        fig, ax = dark_fig(figsize=(7, 4))
        cr = np.linspace(0.5, 0.999, 200)
        vv = [stats.lognorm.ppf(c, s=SIGMA, scale=np.exp(MU) * facteur) for c in cr]
        ax.plot(cr, vv, '-', color='#00c9a7', lw=2)
        ax.fill_between(cr, vv, alpha=0.15, color='#00c9a7')
        for c_, clr_ in [(0.90, '#2a9d8f'), (0.95, '#f4a261'), (0.99, '#e63946')]:
            v_ = stats.lognorm.ppf(c_, s=SIGMA, scale=np.exp(MU) * facteur)
            ax.axvline(c_, color=clr_, ls='--', lw=1, alpha=0.7)
            ax.axhline(v_, color=clr_, ls='--', lw=1, alpha=0.7)
        ax.set_xlabel("Niveau de confiance")
        ax.set_ylabel("VaR — Seuil I(t)")
        ax.set_title(f"Courbe VaR Sanitaire (facteur ×{facteur})", fontsize=10)
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown(f"Monte Carlo — 50 000 scénarios (facteur ×{facteur})")
        np.random.seed(42)
        I_mc = stats.lognorm.rvs(s=SIGMA, scale=np.exp(MU) * facteur, size=50000)
        caps = [20000, 30000, 40000, 50000, 60000, 70000]
        risqs = [np.mean(I_mc > c) for c in caps]

        mc_df = pd.DataFrame({
            "Capacité": [f"{c:,}" for c in caps],
            "P(saturation)": [f"{r:.2%}" for r in risqs],
            "Verdict": ["🔴 Critique" if r > 0.20 else "🟠 Attention" if r > 0.10 else "🟢 OK" for r in risqs]
        })
        st.dataframe(mc_df, hide_index=True, use_container_width=True)

        fig, ax = dark_fig(figsize=(7, 4))
        ax.hist(I_mc, bins=80, density=True, alpha=0.55, color='#457b9d', edgecolor='none')
        ax.axvline(capacite, color='#e63946', ls='--', lw=2, label=f"Capacité choisie ({capacite:,})")
        risk_cap = np.mean(I_mc > capacite)
        ax.set_xlabel("I(t) simulé")
        ax.set_ylabel("Densité")
        ax.set_title(f"MC — Risque saturation = {risk_cap:.1%} (facteur ×{facteur})", fontsize=10)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# =============================================================================
# TAB 5 : Stress Test
# =============================================================================
with tabs[5]:
    st.markdown('<div class="sec-title"> Stress Test — Scénarios de crise</div>', unsafe_allow_html=True)

    scenarios_stress = {
        "Scénario de base": (1.0, "#2a9d8f"),
        "Épidémie modérée (×1.5)": (1.5, "#457b9d"),
        "Épidémie forte (×2)": (2.0, "#f4a261"),
        "Crise majeure (×3)": (3.0, "#e88c0a"),
        "Catastrophe (×5)": (5.0, "#e63946"),
        "Extrême (×10)": (10.0, "#7f0000"),
    }

    rows = []
    for nom, (f, _) in scenarios_stress.items():
        r = 1 - stats.lognorm.cdf(capacite, s=SIGMA, scale=np.exp(MU) * f)
        r_rel = r / risque_dynamique(capacite, 1.0) if risque_dynamique(capacite, 1.0) > 0 else float('inf')
        q50 = stats.lognorm.ppf(0.50, s=SIGMA, scale=np.exp(MU) * f)
        q95 = stats.lognorm.ppf(0.95, s=SIGMA, scale=np.exp(MU) * f)
        rows.append({
            "Scénario": nom,
            "Facteur": f"×{f}",
            f"Risque (cap={capacite:,})": f"{r:.2%}",
            "Risque relatif": f"×{r_rel:.1f}",
            "Médiane I(t)": f"{q50:.0f}",
            "Q95 I(t)": f"{q95:.0f}"
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = dark_fig(figsize=(7, 5))
        noms = [r["Scénario"] for r in rows]
        risques = [float(r[f"Risque (cap={capacite:,})"].strip('%')) / 100 for r in rows]
        couleurs = [v[1] for v in scenarios_stress.values()]
        bars = ax.barh(noms, risques, color=couleurs, edgecolor='#334155', height=0.55)
        for bar, r_ in zip(bars, risques):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{r_:.1%}", va='center', fontsize=9)
        ax.axvline(0.05, color='#f4a261', ls=':', lw=1.5, alpha=0.7, label='Risque 5%')
        ax.axvline(0.10, color='#e63946', ls=':', lw=1.5, alpha=0.7, label='Risque 10%')
        ax.set_xlabel("Probabilité de saturation")
        ax.set_title(f"Risque par scénario (capacité = {capacite:,})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(" Heatmap du risque")
        caps_hm = [20000, 30000, 40000, 50000, 60000]
        facteurs_hm = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        mat = []
        for f in facteurs_hm:
            row_hm = [1 - stats.lognorm.cdf(c, s=SIGMA, scale=np.exp(MU) * f) for c in caps_hm]
            mat.append(row_hm)

        fig, ax = dark_fig(figsize=(7, 5))
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.5)
        ax.set_xticks(range(len(caps_hm)))
        ax.set_xticklabels([f"{c//1000}k" for c in caps_hm], fontsize=8)
        ax.set_yticks(range(len(facteurs_hm)))
        ax.set_yticklabels([f"×{f}" for f in facteurs_hm], fontsize=8)
        ax.set_xlabel("Capacité (lits)")
        ax.set_ylabel("Facteur de crise")
        ax.set_title("Heatmap probabilité de saturation", fontsize=10)
        for i in range(len(facteurs_hm)):
            for j in range(len(caps_hm)):
                ax.text(j, i, f"{mat[i][j]:.0%}", ha='center', va='center',
                        fontsize=8, color='white' if mat[i][j] > 0.25 else 'black')
        plt.colorbar(im, ax=ax, label='Probabilité')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# =============================================================================
# TAB 6 : Seuils d'alerte
# =============================================================================
with tabs[6]:
    st.markdown('<div class="sec-title"> Seuils d\'alerte opérationnels</div>', unsafe_allow_html=True)

    seuils_dict = {
        "🟢 Vigilance": (S75, "Surveillance normale", "#2a9d8f"),
        "🟡 Pré-alerte": (S90, "Renforcement équipes", "#f4a261"),
        "🔴 Alerte": (S95, "Plan Blanc", "#e63946"),
        "⚫ Crise": (S99, "Cellule crise", "#888")
    }

    cols_seuils = st.columns(4)
    for idx, (nom, (seuil, usage, clr)) in enumerate(seuils_dict.items()):
        with cols_seuils[idx]:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{clr}40">
                <span class="kpi-val" style="color:{clr}">{seuil:.0f}</span>
                <div class="kpi-lbl">{nom}</div>
                <div class="kpi-sub">{usage}</div>
            </div>""", unsafe_allow_html=True)

    if df_res is not None and COL_I:
        n = len(I_all)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(" Vérification sur données réelles")
            verif_rows = []
            for nom, (seuil, usage, clr) in seuils_dict.items():
                nb = int(np.sum(I_all > seuil))
                verif_rows.append({
                    "Niveau": nom,
                    "Seuil": f"{seuil:.0f}",
                    "Observé": f"{nb}/{n}",
                    "Fréq. réelle": f"{nb/n:.1%}"
                })
            st.dataframe(pd.DataFrame(verif_rows), hide_index=True, use_container_width=True)

        with col2:
            st.markdown(" Chronologie des dépassements")
            fig, ax = dark_fig(figsize=(7, 4))
            ax.plot(df_res['date'], I_all, '-', color='#a8dadc', lw=1.2, alpha=0.8, label='I(t)')
            for nom, (seuil, usage, clr) in seuils_dict.items():
                ax.axhline(seuil, color=clr, ls='--', lw=1.5, label=f"{nom} ({seuil:.0f})", alpha=0.85)
            ax.set_ylabel("I(t) / 100k")
            ax.set_title("Historique vs seuils d'alerte", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(f"""
        <div style="background:rgba(17,34,64,.8); border:1px solid {niv_col}55;
                    border-radius:8px; padding:.8rem; margin-top:.8rem; text-align:center;">
            <span style="font-size:1.1rem; color:{niv_col}; font-weight:700;">
                 Situation actuelle : I(t) = {I_cur:.0f} → {niv_txt}
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(" Chargez `resultats_sirs_.csv` pour la vérification.")

# =============================================================================
# TAB 7 : What if (sans Météo)
# =============================================================================
with tabs[7]:
    st.markdown('<div class="sec-title"> Scénarios What If</div>', unsafe_allow_html=True)

    r_ref = risque_dynamique(capacite, 1.0)

    whatif_scenarios = [
        (" Situation actuelle", r_ref, "#457b9d"),
        (" Renforcement capacité (+20%)", risque_dynamique(capacite * 1.2, 1.0), "#2a9d8f"),
        (" Mesures réduction I(t) (-30%)", risque_dynamique(capacite, 0.7), "#00c9a7"),
        (" Nouveau variant (×2)", risque_dynamique(capacite, 2.0), "#f4a261"),
        (" Pire scénario (×4)", risque_dynamique(capacite, 4.0), "#e63946"),
    ]

    wif_df = pd.DataFrame({
        "Scénario": [w[0] for w in whatif_scenarios],
        "Risque": [f"{w[1]:.2%}" for w in whatif_scenarios],
        "Δ vs base": [f"{(w[1] - r_ref) / max(r_ref, 1e-6) * 100:+.0f}%" if i > 0 else "—"
                      for i, w in enumerate(whatif_scenarios)]
    })
    st.dataframe(wif_df, hide_index=True, use_container_width=True)

    fig, ax = dark_fig(figsize=(7, 4))
    bars = ax.barh([w[0] for w in whatif_scenarios], [w[1] for w in whatif_scenarios],
                   color=[w[2] for w in whatif_scenarios], edgecolor='#334155', height=0.55)
    for bar, w in zip(bars, whatif_scenarios):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{w[1]:.1%}", va='center', fontsize=9)
    ax.axvline(r_ref, color='white', ls='--', lw=1.5, alpha=0.5, label='Référence')
    ax.set_xlabel("Probabilité de saturation")
    ax.set_title(f"What If (capacité = {capacite:,})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='x')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()





# =============================================================================
# NOUVEL ONGLET : SIMULATIONS
# =============================================================================
with st.expander(" Simulations interactives - Cliquez pour explorer"):
    st.markdown("###  Simulateur d'impact des décisions")
    
    # Paramètres de base
    I_base = np.exp(MU)  # I(t) de référence
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####  Ressources hospitalières")
        augmentation_lits = st.slider(
            "Augmentation de la capacité (%)",
            0, 100, 20, 10,
            help="+20% = ajout de 10 000 lits sur 50 000"
        )
        
        st.markdown("####  Mesures barrières")
        reduction_masques = st.slider(
            "Réduction de I(t) par les mesures (%)",
            0, 50, 30, 5,
            help="Port du masque, distanciation, télétravail"
        )
    
    with col2:
        st.markdown("####  Vaccination")
        couverture_vaccinale = st.slider(
            "Couverture vaccinale (%)",
            0, 95, 75, 5,
            help="Pourcentage de la population vaccinée"
        )
        efficacite_vaccin = 0.85  # 85% d'efficacité
        reduction_vaccin = couverture_vaccinale / 100 * efficacite_vaccin
        
        st.markdown("####  Nouveau variant")
        facteur_variant = st.slider(
            "Contagiosité du variant (multiplicateur)",
            1.0, 4.0, 1.0, 0.5,
            help="x1 = souche originale, x2 = Delta, x2.5 = Omicron"
        )
    
    # ============================================
    # CALCUL DES IMPACTS
    # ============================================
    
    # Impact des mesures barrières
    I_mesures = I_base * (1 - reduction_masques / 100)
    
    # Impact de la vaccination
    I_vaccin = I_base * (1 - reduction_vaccin * 0.5)  # la vaccination réduit I(t)
    
    # Impact du variant
    I_variant = I_base * facteur_variant
    
    # Impact combiné
    I_combine = I_base * (1 - reduction_masques / 100) * (1 - reduction_vaccin * 0.5) * facteur_variant
    
    # Calcul des risques (capacité de référence = 50 000 lits)
    cap_ref = 50000
    risque_base = 1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_base)
    risque_mesures = 1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_mesures)
    risque_vaccin = 1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_vaccin)
    risque_variant = 1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_variant)
    risque_combine = 1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_combine)
    
    # ============================================
    # TABLEAU COMPARATIF
    # ============================================
    st.markdown("###  Impact sur le risque de saturation")
    
    df_simulations = pd.DataFrame({
        "Scénario": [
            " Situation de référence",
            " Mesures barrières",
            " Vaccination (75%)",
            " Nouveau variant",
            " Combinaison (mesures + vaccin)",
            " Pire scénario (variant ×4)"
        ],
        "I(t) estimé": [
            f"{I_base:.0f}",
            f"{I_mesures:.0f}",
            f"{I_vaccin:.0f}",
            f"{I_variant:.0f}",
            f"{I_combine:.0f}",
            f"{I_base * 4:.0f}"
        ],
        "Risque de saturation": [
            f"{risque_base:.2%}",
            f"{risque_mesures:.2%}",
            f"{risque_vaccin:.2%}",
            f"{risque_variant:.2%}",
            f"{risque_combine:.2%}",
            f"{1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_base * 4):.2%}"
        ],
        "Gain / Perte": [
            "—",
            f"-{(risque_base - risque_mesures) / risque_base * 100:.0f}%",
            f"-{(risque_base - risque_vaccin) / risque_base * 100:.0f}%",
            f"+{(risque_variant - risque_base) / risque_base * 100:.0f}%",
            f"-{(risque_base - risque_combine) / risque_base * 100:.0f}%",
            f"+{((1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_base * 4) - risque_base) / risque_base * 100):.0f}%"
        ]
    })
    
    st.dataframe(df_simulations, hide_index=True, use_container_width=True)
    
    # ============================================
    # GRAPHIQUE COMPARATIF
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = df_simulations["Scénario"].tolist()
    risques = [
        risque_base, risque_mesures, risque_vaccin, 
        risque_variant, risque_combine,
        1 - stats.lognorm.cdf(cap_ref, s=SIGMA, scale=I_base * 4)
    ]
    
    couleurs = ['#2a9d8f', '#457b9d', '#00c9a7', '#f4a261', '#e63946', '#7f0000']
    bars = ax.barh(scenarios, risques, color=couleurs, edgecolor='#334155', height=0.6)
    
    for bar, r in zip(bars, risques):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{r:.1%}", va='center', fontsize=10, fontweight='bold')
    
    ax.axvline(risque_base, color='white', ls='--', lw=2, alpha=0.7, label='Référence')
    ax.axvline(0.05, color='orange', ls=':', lw=1.5, alpha=0.7, label='Seuil 5%')
    ax.axvline(0.10, color='red', ls=':', lw=1.5, alpha=0.7, label='Seuil 10%')
    
    ax.set_xlabel("Probabilité de saturation", fontsize=12)
    ax.set_title("Impact des scénarios sur le risque", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # ============================================
    # INTERPRÉTATION
    # ============================================
    st.markdown("###  Interprétation")
    
    meilleur_gain = min(risques[1:5])
    if meilleur_gain == risque_mesures:
        meilleure_mesure = "les mesures barrières"
    elif meilleur_gain == risque_vaccin:
        meilleure_mesure = "la vaccination"
    else:
        meilleure_mesure = "la combinaison"
    
    st.info(f"""
    Enseignements clés :
    
    1. {meilleure_mesure} est la mesure la plus efficace avec une réduction du risque de {(risque_base - meilleur_gain) / risque_base * 100:.0f}%.
    
    2. La vaccination seule réduit le risque de {(risque_base - risque_vaccin) / risque_base * 100:.0f}%.
    
    3. Les mesures barrières seules réduisent le risque de {(risque_base - risque_mesures) / risque_base * 100:.0f}%.
    
    4. L'association (mesures + vaccination) est encore plus efficace : réduction de {(risque_base - risque_combine) / risque_base * 100:.0f}%.
    
    5. Un nouveau variant multiplierait le risque par {(risque_variant / risque_base):.1f}.
    
     Recommandation : Privilégier la prévention (vaccination + mesures barrières) plutôt que l'augmentation des lits, qui a un effet plus limité.
    """)








# =============================================================================
# TAB 8 : Bootstrap
# =============================================================================
with tabs[8]:
    st.markdown('<div class="sec-title"> Bootstrap — Intervalles de confiance 95%</div>', unsafe_allow_html=True)

    st.info("Méthode : 50 rééchantillonnages avec remise + points de départ variés. "
            "Chaque optimisation explore l'espace des paramètres indépendamment.")

    bs_rows = []
    for param, vals in BOOTSTRAP.items():
        bs_rows.append({
            "Paramètre": param,
            "Estimation": f"{vals['moy']:.4f}",
            "Écart-type": f"{vals['std']:.4f}",
            "IC 95% [inf — sup]": f"[{vals['ci'][0]:.4f} — {vals['ci'][1]:.4f}]",
            "CV (%)": f"{vals['std'] / vals['moy'] * 100:.1f}%",
            "Stabilité": "" if vals['std'] / vals['moy'] < 0.15 else ""
        })
    st.dataframe(pd.DataFrame(bs_rows), hide_index=True, use_container_width=True)

    st.markdown(" Distributions bootstrap simulées")
    np.random.seed(42)
    fig, axes = dark_fig(2, 3, figsize=(14, 7))
    for idx, (param, vals) in enumerate(BOOTSTRAP.items()):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        sim = np.clip(np.random.normal(vals['moy'], vals['std'], 1000), 0.001, None)
        ax.hist(sim, bins=30, color=C[idx % len(C)], edgecolor='#334155', alpha=0.75)
        ax.axvline(vals['moy'], color='white', ls='--', lw=2, label=f"Moy = {vals['moy']:.4f}")
        ax.axvline(vals['ci'][0], color='#8892b0', ls=':', lw=1.5, label='IC 95%')
        ax.axvline(vals['ci'][1], color='#8892b0', ls=':', lw=1.5)
        ax.set_title(f"Bootstrap {param}", fontsize=10)
        ax.set_xlabel(param)
        ax.legend(fontsize=7)
        ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
     Interprétation :
    - Distributions resserrées → optimisation convergente et stable
    - α ∈ [0.262 ; 0.536] avec 95% de confiance → entre 26% et 54% des infectés vont aux urgences
    - R₀ ∈ [5.02 ; 7.84] → cohérent avec la littérature COVID-19
    """)

# =============================================================================
# TAB 9 : Données brutes
# =============================================================================
with tabs[9]:
    st.markdown('<div class="sec-title"> Exploration des données brutes</div>', unsafe_allow_html=True)

    ds_choice = st.radio(
        "Choisir le dataset :",
        ["Résultats SIRS", "Données brutes COVID-19"],
        horizontal=True
    )

    df_show = None
    if ds_choice == "Résultats SIRS":
        df_show = df_res
    else:
        if df_raw is not None:
            dep_fc = [25, 39, 70, 90]
            if 'Département Code' in df_raw.columns:
                df_show = df_raw[df_raw['Département Code'].isin(dep_fc)]
            else:
                df_show = df_raw
        else:
            st.error(f" {err_raw}")

    if df_show is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric(" Lignes", f"{len(df_show):,}")
        c2.metric(" Colonnes", f"{len(df_show.columns)}")
        if 'date' in df_show.columns:
            c3.metric(" Période", f"{df_show['date'].min().strftime('%m/%Y')} → {df_show['date'].max().strftime('%m/%Y')}")

        st.dataframe(df_show.head(100), use_container_width=True)

        st.markdown(" Statistiques descriptives")
        st.dataframe(df_show.describe().round(2), use_container_width=True)

        csv_dl = df_show.to_csv(index=False)
        st.download_button(
            " Télécharger ce dataset (.csv)",
            csv_dl,
            f"gesica_{ds_choice.lower().replace(' ', '_')}.csv",
            "text/csv"
        )
        
        
# =============================================================================
# TAB 10 : À PROPOS
# =============================================================================
with tabs[10]:
    st.markdown('<div class="sec-title">ℹ️ À propos</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Dashboard Gesica / Interreg
    
    **Auteur** : Nassir Ousmane — M1 Ingénierie de la Santé
    
    **Laboratoire** : SINeRGIE Besançon
    
    **Période** : 2020-2025
    
    **Données** : OScour (passages aux urgences pour COVID-19)
    
    **Modèle** : SIRS calibré sur la Franche-Comté
    
    ---
    
    ### Validation du modèle
    
    | Métrique | Valeur |
    |----------|--------|
    | R² (entraînement) | 0.5906 |
    | R² (test) | 0.7018 |
    | RMSE | 1 956 |
    | MAE | 1 189 |
    | Bootstrap | 50 échantillons |
    
    ---
    
    ### Loi Log-Normale de I(t)
    
    - **μ** = 7.3189
    - **σ** = 1.4358
    - **p-value KS** = 0.605 → loi acceptée
    
    ---
    
    ### Liens
    
    - [Code source sur GitHub](https://github.com/nassiro2023-cmd/dashboard-gesica)
    - [Projet Gesica](https://www.interreg-francesuisse.eu/projets/gesica)
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <strong>Dashboard Gesica / Interreg</strong> ·
    Nassir Ousmane — M1 Ingénierie de la Santé IA & Systèmes de Santé ·
    Laboratoire SINeRGIE Besançon · 2025<br>
    <small>Données OScour · Modèle SIRS calibré sur Franche-Comté 2020–2025 ·
    Bootstrap 50 échantillons · Validation croisée chronologique ·
    Loi Log-Normale acceptée (p-value KS = 0.605)</small>
</div>
""", unsafe_allow_html=True)














