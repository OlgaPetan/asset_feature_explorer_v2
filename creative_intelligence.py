"""
Asset Intelligence — Asset Feature Explorer
The Coca-Cola Company
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Asset Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── palette ───────────────────────────────────────────────────────────────────
RED    = "#E8002D"
DARK   = "#1C1C1C"
MID    = "#6A6660"
LIGHT  = "#FAFAF8"
BLUE   = "#2D5BE3"
GREEN  = "#2A8050"
AMBER  = "#906820"
PURPLE = "#7030A0"
BORDER = "#EAE8E2"

M_COLOR = {
    "Attention_T2B":             BLUE,
    "Persuasion_T2B":            RED,
    "Likeability_Love_Like_T2B": GREEN,
    "Experience_Recall_T2B":     "#B45309",
    "Brand_Linkage_T2B":         "#0E7490",
    "Uniqueness_T2B":            "#6D28D9",
    "Shareability_T2B":          "#065F46",
    "Tiredness_T2B":             "#9F1239",
}
M_LABEL = {
    "Attention_T2B":             "Attention",
    "Persuasion_T2B":            "Persuasion",
    "Likeability_Love_Like_T2B": "Likeability",
    "Experience_Recall_T2B":     "Experience Recall",
    "Brand_Linkage_T2B":         "Brand Linkage",
    "Uniqueness_T2B":            "Uniqueness",
    "Shareability_T2B":          "Shareability",
    "Tiredness_T2B":             "Tiredness",
}
NEW_METRICS = ["Experience_Recall_T2B","Brand_Linkage_T2B","Uniqueness_T2B",
               "Shareability_T2B","Tiredness_T2B"]
LOWER_BETTER = {"Tiredness_T2B"}   # 5-pt scale, 5 = most tired; T2B = share answering 4-5 → higher is worse
LABEL_TO_MC = {v: k for k, v in M_LABEL.items()}
METRIC_NAMES = list(M_LABEL.values())
RULE_STYLE = {
    "Conflict":           ("warn-conflict",     "⚡"),
    "Heterogeneity":      ("warn-heterogeneity","⚠"),
    "Boundary Condition": ("warn-boundary",     "◈"),
    "Opportunity":        ("warn-opportunity",  "◎"),
    "Outlier":            ("warn-outlier",      "↗"),
    "Consensus":          ("warn-consensus",    "✓"),
    "Anti-pattern":       ("warn-antipattern",  "✕"),
    "Insight":            ("warn-insight",      "ℹ"),
}
RULE_ACTION = {
    "Conflict":           "Helps one metric, hurts another — decide which metric this brief is for.",
    "Heterogeneity":      "Average hides a reversal here — do not brief from the overall number for this scope.",
    "Boundary Condition": "Works in one format only — brief it for that format.",
    "Opportunity":        "Works but is rarely used — test more assets with it.",
    "Outlier":            "This scope behaves unlike its peers — check before generalising.",
    "Consensus":          "Positive on Attention, Persuasion and Likeability — safe to brief.",
    "Anti-pattern":       "Negative on Attention, Persuasion and Likeability here — brief against it.",
    "Insight":            "Tested finding from the Insight Catalog.",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

html,body,[class*="css"]{font-family:'Source Sans 3',sans-serif;font-size:15px;background:#FAFAF8;color:#1C1C1C;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2.8rem 4rem 2.8rem!important;max-width:1440px;}

.topbar{background:#E8002D;margin:0 -2.8rem 0 -2.8rem;padding:.6rem 2.8rem;display:flex;align-items:center;justify-content:space-between;}
.topbar-left{display:flex;align-items:center;gap:.9rem;}
.topbar-logo{font-family:'Merriweather',serif;font-size:1.05rem;font-weight:400;color:#fff;}
.topbar-pipe{width:1px;height:16px;background:rgba(255,255,255,.3);}
.topbar-sub{font-size:.75rem;color:rgba(255,255,255,.65);letter-spacing:.1em;text-transform:uppercase;}

.hero{padding:2rem 0 1.5rem 0;border-bottom:1px solid #EAE8E2;margin-bottom:2rem;}
.hero-eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:#E8002D;margin-bottom:.5rem;}
.hero-title{font-family:'Merriweather',serif;font-size:2.4rem;font-weight:300;line-height:1.2;color:#1C1C1C;margin-bottom:.7rem;}
.hero-sub{font-size:.95rem;color:#7A7670;font-weight:300;max-width:620px;line-height:1.75;}

.sec-label{font-size:.72rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#6A6660;margin-bottom:.8rem;}

.scope-chip{display:inline-flex;align-items:center;background:#fff0f0;border:1px solid #f0c0c0;border-radius:3px;padding:.2rem .7rem;font-size:.82rem;color:#c00020;margin:.18rem .25rem .18rem 0;font-weight:500;}

.kpi-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.1rem 1.3rem;}
.kpi-val{font-family:'Merriweather',serif;font-size:2rem;font-weight:300;line-height:1;}
.kpi-lbl{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#AAA8A0;margin-top:.3rem;}
.kpi-sub{font-size:.78rem;color:#888;margin-top:.22rem;}

.explain-box{background:#F5F3EF;border-left:3px solid #E8002D;border-radius:0 4px 4px 0;padding:.8rem 1rem;font-size:.82rem;color:#555;line-height:1.65;margin-bottom:1rem;}

.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.69rem;font-weight:600;letter-spacing:.03em;}
.b-pos{background:#E8F8EF;color:#2A8050;}.b-neg{background:#FDEDEC;color:#C00020;}
.b-ns{background:#F2F2F2;color:#888;}.b-sig{background:#F3EEFF;color:#7030A0;}
.b-metric{background:#EBF3FB;color:#2D5BE3;}.b-scope{background:#F5F5F5;color:#555;}
.b-hi{background:#EAFAF1;color:#27AE60;}.b-med{background:#FEF9E7;color:#E67E22;}.b-lo{background:#F2F2F2;color:#888;}

.feat-row{display:flex;align-items:center;padding:.55rem 0;border-bottom:1px solid #F5F3EF;}
.feat-name{width:140px;font-size:.88rem;font-weight:600;color:#1C1C1C;flex-shrink:0;}
.feat-bar-wrap{flex:1;display:flex;align-items:center;gap:.5rem;}
.feat-val{width:68px;text-align:right;font-size:.85rem;font-weight:600;flex-shrink:0;}
.feat-sig{width:28px;font-size:.72rem;text-align:center;flex-shrink:0;}
.feat-n{width:52px;font-size:.72rem;color:#AAA;text-align:right;flex-shrink:0;}

.combo-card{background:#fff;border:1px solid #EAE8E2;border-left:3px solid #E8002D;border-radius:0 6px 6px 0;padding:1.6rem 1.8rem;margin-bottom:1.3rem;box-shadow:0 1px 4px rgba(0,0,0,.04);}
.combo-uplift{font-family:'Merriweather',serif;font-size:3.2rem;color:#E8002D;line-height:1;font-weight:300;}
.pill-row{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.9rem;}
.pill{display:inline-flex;align-items:center;gap:.4rem;background:#FAFAF8;border:1px solid #EAE8E2;border-radius:3px;padding:.26rem .75rem;font-size:.82rem;color:#5A5650;}
.pill-n{background:#E8002D;color:#fff;border-radius:2px;width:17px;height:17px;display:inline-flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;flex-shrink:0;}
.pill-gain{color:#2A8050;font-size:.79rem;font-weight:600;}

.wf-row{display:flex;align-items:center;gap:.9rem;padding:.52rem 0;border-bottom:1px solid #F5F3EF;}
.wf-idx{width:20px;font-size:.74rem;color:#CCC8C2;text-align:center;flex-shrink:0;}
.wf-label{flex:1;font-size:.88rem;color:#3A3830;}
.wf-bar{width:140px;flex-shrink:0;}
.wf-n{width:50px;font-size:.73rem;color:#BBB8B2;text-align:right;flex-shrink:0;}
.wf-pp{width:60px;font-size:.86rem;font-weight:600;color:#2A8050;text-align:right;flex-shrink:0;}
.wf-col{font-size:.64rem;letter-spacing:.12em;text-transform:uppercase;color:#CCC8C2;}

.explorer-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.6rem 1.8rem;margin-top:1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.04);}
.explorer-title{font-family:'Merriweather',serif;font-size:1.3rem;font-weight:300;color:#1C1C1C;margin-bottom:.22rem;}
.explorer-sub{font-size:.87rem;color:#AAA89E;line-height:1.65;margin-bottom:1.5rem;}
.feat-group{font-size:.65rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#CCC8C2;margin:1.1rem 0 .65rem 0;padding-bottom:.3rem;border-bottom:1px solid #F0EEE8;}

.scoreboard{display:flex;gap:.9rem;margin-top:1.3rem;flex-wrap:wrap;}
.score-cell{flex:1;background:#FAFAF8;border:1px solid #EAE8E2;border-radius:5px;padding:1rem 1.1rem;text-align:center;min-width:100px;}
.score-cell.hi{border-color:#E8002D;background:#FFF8F8;}
.score-metric{font-size:.67rem;letter-spacing:.14em;text-transform:uppercase;color:#CCC8C2;margin-bottom:.35rem;}
.score-val{font-family:'Merriweather',serif;font-size:2.1rem;font-weight:300;color:#1C1C1C;line-height:1;}
.score-delta{font-size:.82rem;font-weight:500;margin-top:.25rem;}
.d-up{color:#2A8050;}.d-dn{color:#C03030;}.d-flat{color:#CCC8C2;}
.score-n{font-size:.72rem;color:#D8D4CE;margin-top:.2rem;}

.asset-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:.85rem 1rem;margin-bottom:.5rem;}
.asset-name{font-size:.8rem;font-weight:600;color:#1C1C1C;word-break:break-all;}
.asset-meta{font-size:.72rem;color:#AAA89E;margin-top:.15rem;}
.asc{font-size:.7rem;padding:2px 7px;border-radius:3px;font-weight:600;margin-right:.3rem;}
.asc-a{background:#EBF3FB;color:#2D5BE3;}.asc-p{background:#FFF0F0;color:#C00020;}
.asc-l{background:#EAFAF1;color:#2A8050;}.asc-s{background:#F3EEFF;color:#7030A0;}

.insight-strip{display:flex;flex-direction:column;gap:.38rem;margin:.45rem 0;}
.insight-warn{display:flex;align-items:flex-start;gap:.6rem;border-radius:4px;padding:.52rem .82rem;font-size:.82rem;line-height:1.5;border:1px solid;}
.iw-icon{flex-shrink:0;margin-top:.04rem;}.iw-body{flex:1;}
.iw-type{font-size:.64rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.14rem;}
.warn-conflict{background:#FFF0F0;border-color:#F0C0C0;}.warn-conflict .iw-type{color:#C00020;}
.warn-heterogeneity{background:#FFFBF0;border-color:#EDD080;}.warn-heterogeneity .iw-type{color:#906820;}
.warn-boundary{background:#F8F0FF;border-color:#D4B0F0;}.warn-boundary .iw-type{color:#7030A0;}
.warn-opportunity{background:#F0FFF4;border-color:#90D8A8;}.warn-opportunity .iw-type{color:#1A7040;}
.warn-outlier{background:#EBF3FB;border-color:#90B0E8;}.warn-outlier .iw-type{color:#1A3080;}
.warn-consensus{background:#F0FFF4;border-color:#90D8A8;}.warn-consensus .iw-type{color:#1A7040;}
.warn-antipattern{background:#FFF0F0;border-color:#F0C0C0;}.warn-antipattern .iw-type{color:#C00020;}
.warn-insight{background:#FAFAF8;border-color:#E4E0DA;}.warn-insight .iw-type{color:#6A6660;}

.insight-card{background:#fff;border:1px solid #E8E4DC;border-left:4px solid #E8002D;border-radius:6px;padding:12px 16px;margin-bottom:8px;}
.ic-feat{font-weight:600;font-size:.9rem;color:#1C1C1C;}
.ic-text{font-size:.82rem;color:#444;margin-top:3px;line-height:1.5;}
.ic-meta{display:flex;gap:6px;margin-top:7px;flex-wrap:wrap;}

.low-n{background:#FFFBF0;border:1px solid #EDD080;border-radius:4px;padding:.46rem .85rem;font-size:.81rem;color:#906820;margin-top:.6rem;}
hr.div{border:none;border-top:1px solid #EAE8E2;margin:1.8rem 0;}
.footer{display:flex;justify-content:space-between;font-size:.72rem;color:#CCC8C2;padding:1.1rem 0 .7rem 0;border-top:1px solid #EAE8E2;margin-top:2rem;}

.stButton>button{background:#fff!important;border:1px solid #DAD6D0!important;color:#6A6660!important;border-radius:4px!important;font-size:.88rem!important;width:100%!important;}
.stButton>button:hover{border-color:#E8002D!important;color:#E8002D!important;}

.scope-btn-wrap{display:flex;gap:.5rem;margin-bottom:.6rem;}
.scope-btn{display:inline-flex;align-items:center;justify-content:center;padding:.42rem 1.2rem;border-radius:20px;font-size:.82rem;font-weight:500;cursor:pointer;border:1.5px solid #DAD6D0;background:#fff;color:#6A6660;transition:all .15s;white-space:nowrap;}
.scope-btn.active{background:#E8002D;border-color:#E8002D;color:#fff;font-weight:600;}
.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid #EAE8E2;gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#AAA89E;font-size:.9rem;border-radius:0;padding:.52rem 1.3rem;border:none;border-bottom:2px solid transparent;margin-bottom:-1px;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#1C1C1C!important;border-bottom:2px solid #E8002D!important;font-weight:600!important;}
.stSelectbox label{font-size:.8rem!important;color:#AAA89E!important;font-weight:400!important;letter-spacing:.05em!important;text-transform:uppercase!important;}
.stSelectbox>div>div{background:#FAFAF8!important;border:1px solid #E4E0DA!important;border-radius:4px!important;color:#1C1C1C!important;font-size:.9rem!important;}

.page-card{background:#fff;border:1px solid #EAE8E2;border-radius:8px;padding:1.3rem 1.5rem;margin-bottom:.8rem;display:flex;gap:1.2rem;align-items:flex-start;}
.page-card-icon{font-size:1.6rem;line-height:1;flex-shrink:0;margin-top:.1rem;}
.page-card-body{flex:1;}
.page-card-title{font-size:.92rem;font-weight:700;color:#1C1C1C;margin-bottom:.25rem;}
.page-card-phase{display:inline-block;font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;padding:.15rem .55rem;border-radius:20px;margin-bottom:.45rem;}
.phase-what{background:#EBF3FB;color:#2D5BE3;}
.phase-why{background:#EAFAF1;color:#2A8050;}
.phase-sowhat{background:#F3EEFF;color:#7030A0;}
.page-card-desc{font-size:.83rem;color:#555;line-height:1.65;}
.page-card-tip{font-size:.77rem;color:#888;margin-top:.5rem;border-top:1px solid #F5F3EF;padding-top:.45rem;}

.metric-ref-row{display:flex;align-items:flex-start;gap:.7rem;padding:.55rem 0;border-bottom:1px solid #F5F3EF;}
.metric-ref-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:.25rem;}
.metric-ref-name{font-size:.85rem;font-weight:600;color:#1C1C1C;width:160px;flex-shrink:0;}
.metric-ref-desc{font-size:.82rem;color:#555;line-height:1.5;}

/* top navigation (the only st.radio in the app) */
div[role="radiogroup"]{flex-wrap:wrap;gap:.3rem;padding:.55rem 0 .65rem 0;border-bottom:1px solid #EAE8E2;}
div[role="radiogroup"] label{background:#fff;border:1px solid #E4E0DA;border-radius:20px;padding:.28rem .95rem;margin:0!important;cursor:pointer;}
div[role="radiogroup"] label:hover{border-color:#E8002D;}
div[role="radiogroup"] label > div:first-child{display:none;}
div[role="radiogroup"] label p{font-size:.84rem;color:#6A6660;}
div[role="radiogroup"] label:has(input:checked){background:#E8002D;border-color:#E8002D;}
div[role="radiogroup"] label:has(input:checked) p{color:#fff;font-weight:600;}

/* emphasised alerts */
.alert-panel{background:#fff;border:1px solid #F0C0C0;border-top:3px solid #E8002D;border-radius:6px;padding:.9rem 1rem;margin:.4rem 0 1rem 0;}
.alert-panel-title{font-size:.8rem;font-weight:700;color:#C00020;margin-bottom:.5rem;}
.iw-action{font-size:.78rem;color:#1C1C1C;margin-top:.25rem;font-weight:600;}
.iw-scope{font-size:.68rem;color:#888;margin-top:.15rem;}

/* takeaways + notes */
.takeaway{background:#fff;border:1px solid #EAE8E2;border-left:4px solid #2D5BE3;border-radius:0 6px 6px 0;padding:.9rem 1.1rem;margin-bottom:1rem;}
.takeaway-title{font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#2D5BE3;margin-bottom:.45rem;}
.takeaway li{font-size:.86rem;color:#1C1C1C;line-height:1.6;margin-bottom:.2rem;}
.def-note{font-size:.76rem;color:#6A6660;background:#FAFAF8;border:1px dashed #DAD6D0;border-radius:4px;padding:.45rem .75rem;margin:.3rem 0 .8rem 0;line-height:1.55;}
.tired-note{font-size:.8rem;color:#7A1030;background:#FFF0F3;border:1px solid #F3C0CC;border-radius:4px;padding:.55rem .85rem;margin:.3rem 0 .9rem 0;line-height:1.55;}
.b-dir{background:#FFF7E6;color:#906820;}
</style>
""", unsafe_allow_html=True)


# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    with open("precomputed_data.pkl", "rb") as f:
        payload = pickle.load(f)
    results = payload["results"]
    meta    = payload["meta"]
    df      = meta["df"].copy()

    # SCD score
    see  = ["Experience_Recall_T2B_percentile","Brand_Linkage_T2B_percentile","Comprehension_T2B_percentile"]
    conn = ["Likeability_Love_Like_T2B_percentile","Uniqueness_T2B_percentile","Brand_Interest_T2B_percentile"]
    do   = ["Persuasion_T2B_percentile","Shareability_T2B_percentile"]
    df["SCD_score"] = (
        df[[c for c in see  if c in df.columns]].mean(axis=1, skipna=False) * 0.1 +
        df[[c for c in conn if c in df.columns]].mean(axis=1, skipna=False) * 0.3 +
        df[[c for c in do   if c in df.columns]].mean(axis=1, skipna=False) * 0.6
    ).round(4)

    # insight catalog + rulebook
    try:
        catalog  = pd.read_csv("insight_catalog.csv")
        rulebook = pd.read_csv("rulebook.csv")
        for d in [catalog, rulebook]:
            for c in d.select_dtypes(include="object").columns:
                d[c] = d[c].fillna("")
        catalog["evidence_uplift_pp"] = pd.to_numeric(catalog["evidence_uplift_pp"], errors="coerce")
        has_ins = True
    except FileNotFoundError:
        catalog = rulebook = None
        has_ins = False

    try:
        uplift_df = pd.read_csv("uplift_all_scopes.csv")
        uplift_df["uplift_pp"] = pd.to_numeric(uplift_df["uplift_pp"], errors="coerce")
    except FileNotFoundError:
        uplift_df = None

    camp_map = (df[["campaign_sk_id","campaign_display_name","campaign_code"]]
                .drop_duplicates("campaign_sk_id")
                .set_index("campaign_sk_id"))

    # year (from the first usable date-like column)
    df["_year"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for c in ["year","start_date","start_date_sk_id","end_date","end_date_sk_id","test_date","date"]:
        if c not in df.columns: continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > .5 and num.dropna().between(19000101, 21001231).all():
            df["_year"] = (num // 10000).astype("Int64"); break
        if num.notna().mean() > .5 and num.dropna().between(1900, 2100).all():
            df["_year"] = num.astype("Int64"); break
        dt = pd.to_datetime(df[c], errors="coerce")
        if dt.notna().mean() > .5:
            df["_year"] = dt.dt.year.astype("Int64"); break

    # TCCC vs competitor flag
    if "is_tccc_brand" in df.columns:
        df["_tccc"] = df["is_tccc_brand"].astype(str).str.strip().str.lower().isin(
            ["1","1.0","true","t","yes","y"])

    return df, results, meta, catalog, rulebook, uplift_df, has_ins, camp_map

try:
    df_full, results, meta, catalog, rulebook, uplift_df, has_ins, camp_map = load_all()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}"); st.stop()

METRICS      = meta["metrics"]   # {col: label}
FEAT_LABEL   = meta["feat_label"]
BIN_FEATS    = meta["binary_feats"]
CAT_FEATS    = meta["cat_feats"]
ALL_FEATS    = [
    "animals_and_pets_presence","animatics_cartoons_presence","food_presence",
    "human_presence","outdoors","indoors","product_presence",
    "color_contrast_cat","text_color_contrast_cat","color_spectrum",
    "occasions","passion_point","moments","music_style","seasonal",
]
FEAT_DESC = {
    "animals_and_pets_presence":  "Animals or pets in the ad create warmth and emotional connection, though they can distract from the brand.",
    "animatics_cartoons_presence":"Animated or cartoon-style creative allows flexibility and stands out in feed, but may reduce realism.",
    "food_presence":              "Food or drink visibly featured. Activates appetite and occasion-based associations.",
    "human_presence":             "A person appears in the ad. Creates relatability and emotional resonance with viewers.",
    "outdoors":                   "Ad is set in an outdoor environment. Conveys freedom, lifestyle, and occasion fit.",
    "indoors":                    "Ad is set indoors. Conveys comfort and occasion context (meals, socialising at home).",
    "product_presence":           "The Coca-Cola product is visible. Directly drives brand linkage and aided recall.",
    "color_contrast_cat":         "Level of color contrast in the ad. Higher contrast drives attention but must balance with brand aesthetics.",
    "text_color_contrast_cat":    "Legibility of on-screen text. Poor contrast reduces comprehension and key message effectiveness.",
    "color_spectrum":             "Dominant colour palette direction (warm vs cool). Warm = energy/joy; cool = refreshment.",
    "tone":                       "Emotional tone of the ad (friendly, inspiring, humorous…). Alignment with brand values drives connection.",
    "design_style":               "Overall visual style (photographic, illustrated, typographic…). Affects perceived quality and brand fit.",
    "occasions":                  "Consumption occasion targeted (Meals, Festive, Sport…). Occasion relevance drives purchase intent.",
    "passion_point":              "Consumer interest or lifestyle passion connected to (Music, Sport, Food…). Drives engagement and shareability.",
    "moments":                    "Specific social moment depicted. Moment relevance drives emotional connection.",
    "music_style":                "Genre of music used. Affects mood, attention, and cultural resonance.",
    "seasonal":                   "Tied to a season or holiday. Boosts short-term relevance but limits asset longevity.",
    "emotions":                   "Emotions detected in the creative. Emotional congruence with the audience drives likeability and recall.",
    "food":                       "Specific food items depicted. Provides context for occasion and appetite-driven targeting.",
    "products":                   "Coca-Cola product variants shown. Product specificity drives brand linkage.",
    "intrinsic_elements":         "Core visual brand elements (bottle shape, logo placement, brand colours).",
    "additional_elements_and_product_placement": "Secondary objects in the scene besides the hero product (props, packaging, signage) and how the product is placed in it.",
    "most_frequently_used_word_in_creative": "The dominant text/copy message. Affects comprehension and key message recall.",
}
SCOPE_COL = {"ou":"operating_unit_code","category":"category",
             "brand":"brand_name","market":"country_name"}  # market kept in SCOPE_COL for backward compat with alerts
ALL_METRICS = list(dict.fromkeys(list(METRICS.keys()) + NEW_METRICS))


# ── html helpers ──────────────────────────────────────────────────────────────
def badge(txt, cls): return f'<span class="badge {cls}">{txt}</span>'
def sig_badge(s):
    if s in ("***","**","*"): return badge(s,"b-sig")
    return badge("ns","b-ns")
REL_BADGE = {"High": "b-hi", "Medium": "b-med", "Directional": "b-dir"}
def rel_label(sig, n1, n0):
    """Reliability: High = ** or *** with 50+ assets each side · Medium = significant with 20+ each side ·
    Directional = everything else (not significant or small sample)."""
    if sig in ("**","***") and n1 >= 50 and n0 >= 50: return "High"
    if sig in ("*","**","***") and n1 >= 20 and n0 >= 20: return "Medium"
    return "Directional"
def rel_from_conf(c):
    return {"high": "High", "medium": "Medium"}.get(str(c).lower(), "Directional")
def rel_badge(lbl): return badge(lbl, REL_BADGE.get(lbl, "b-dir"))
def conf_badge(c): return rel_badge(rel_from_conf(c))

def lower_better(mc): return mc in LOWER_BETTER
def is_good(v, mc):
    """True when the uplift v is good news for metric mc (Tiredness: negative is good)."""
    return (v < 0) if lower_better(mc) else (v > 0)
def ucol(v, mc, thr=0.0):
    if v is None or pd.isna(v) or abs(v) <= thr: return AMBER
    return GREEN if is_good(v, mc) else RED
def mc_of(label): return LABEL_TO_MC.get(label, label)

def tired_note(mc):
    if lower_better(mc):
        st.markdown('<div class="tired-note"><strong>Reading Tiredness:</strong> asked on a 5-point scale '
                    '(5 = most tired, 1 = least tired). The score is the % of viewers answering 4–5, so '
                    '<strong>higher = more fatigue = worse</strong>. A <strong>positive uplift is bad</strong> '
                    '(the feature makes the ad more tiring, shown red); a negative uplift is good (shown green).</div>',
                    unsafe_allow_html=True)

def how_to(html):
    with st.expander("How to read this page", expanded=False):
        st.markdown(f'<div style="font-size:.84rem;color:#444;line-height:1.7">{html}</div>', unsafe_allow_html=True)

def dl(df, name, key, label="⬇ Download CSV"):
    if df is None or len(df) == 0: return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{name}.csv", mime="text/csv", key=f"dl_{key}")

def rule_matches_metric(rmetric, mc):
    """Keep a rulebook row only if it is about the selected metric."""
    if not mc: return True
    rm = str(rmetric or "")
    if rm in ("", "All metrics"):     # "all metrics" rules cover Attention, Persuasion, Likeability
        return mc in ("Attention_T2B","Persuasion_T2B","Likeability_Love_Like_T2B")
    return mc in rm or M_LABEL.get(mc, "~") in rm

def render_alerts(items, max_items=6, title=None):
    if not items: return
    import re
    def highlight_pp(text):
        def repl(m):
            val = m.group(0)
            try:
                num = float(re.sub(r"[^\d.\-]","", val.replace("pp","")))
                color = "#2A8050" if num >= 0 else "#C00020"
            except: color = "#1C1C1C"
            return f'<strong style="color:{color};font-size:.88rem">{val}</strong>'
        return re.sub(r"[+\-]?\d+\.?\d*pp", repl, text)

    def expand_metric_abbrevs(text):
        """Replace Att/Pers/Like shorthand with full metric names."""
        text = re.sub(r'\bAtt\b', 'Attention', text)
        text = re.sub(r'\bPers\b', 'Persuasion', text)
        text = re.sub(r'\bLike\b', 'Likeability', text)
        text = re.sub(r'\bRec\b', 'Recall', text)
        text = re.sub(r'\bShare\b', 'Shareability', text)
        text = re.sub(r'\bTired\b', 'Tiredness', text)
        text = re.sub(r'\bLink\b', 'Brand Linkage', text)
        text = re.sub(r'\bUniq\b', 'Uniqueness', text)
        text = re.sub(r'all 3 metrics', 'all metrics (Attention, Persuasion, Likeability)', text)
        return text
    html = '<div class="alert-panel">' if title else ''
    if title: html += f'<div class="alert-panel-title">{title}</div>'
    html += '<div class="insight-strip">'
    for it in items[:max_items]:
        css, icon = RULE_STYLE.get(it["type"], ("warn-insight","ℹ"))
        display_text = expand_metric_abbrevs(highlight_pp(it["text"]))
        action = RULE_ACTION.get(it["type"], "")
        scope = it.get("scope", "")
        html += (f'<div class="insight-warn {css}">'
                 f'<div class="iw-icon">{icon}</div>'
                 f'<div class="iw-body">'
                 f'<div class="iw-type">{it["type"]}</div>'
                 f'<div style="font-size:.82rem;color:#333">{display_text}</div>'
                 + (f'<div class="iw-action">→ {action}</div>' if action else '')
                 + (f'<div class="iw-scope">Applies to: {scope}</div>' if scope else '')
                 + '</div></div>')
    html += "</div>" + ("</div>" if title else "")
    st.markdown(html, unsafe_allow_html=True)

SMAP = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
def _scope_txt(scope, val):
    return "Global — all assets, not specific to your filters" if scope == "Global" else f"{scope} = {val}"

def get_scope_alerts(scope_filters, mc=None, max_items=6):
    if not has_ins or rulebook is None: return []
    rb = rulebook[rulebook["metric"].apply(lambda m: rule_matches_metric(m, mc))] if mc else rulebook
    mask = pd.Series(False, index=rb.index)
    for t,v in scope_filters:
        mask |= ((rb["scope"]==SMAP.get(t,t)) & (rb["scope_value"]==v))
    scoped = rb[mask]
    glob = rb[(rb["scope"]=="Global") & (rb["rule_type"].isin(["Conflict","Anti-pattern","Consensus","Opportunity"]))]
    sev = lambda s: s.map({"high":0,"medium":1,"low":2})
    out = pd.concat([scoped.sort_values("severity", key=sev), glob.sort_values("severity", key=sev)])
    return [{"type":r["rule_type"],"text":r["text"],"scope":_scope_txt(r["scope"],r["scope_value"])}
            for _,r in out.head(max_items).iterrows()]

def get_feature_alerts(feature, scope_filters, metric_col, max_items=3):
    if not has_ins or rulebook is None or catalog is None: return []
    ml = M_LABEL.get(metric_col, metric_col)
    items = []
    rb = rulebook[(rulebook["feature"]==feature) &
                  (rulebook["metric"].apply(lambda m: rule_matches_metric(m, metric_col)))]
    m_s = pd.Series(False, index=rb.index)
    for t,v in scope_filters: m_s |= (rb["scope"]==SMAP.get(t,t)) & (rb["scope_value"]==v)
    rb_s = rb[m_s]
    sev = lambda s: s.map({"high":0,"medium":1,"low":2})
    rb = pd.concat([rb_s.sort_values("severity", key=sev),
                    rb[rb["scope"]=="Global"].sort_values("severity", key=sev)]).drop_duplicates()
    for _,r in rb.head(max_items).iterrows():
        items.append({"type":r["rule_type"],"text":r["text"],"scope":_scope_txt(r["scope"],r["scope_value"])})
    if len(items) < max_items:
        cat = catalog[(catalog["feature"]==feature) &
                      (catalog["metric_display"]==ml) &
                      (catalog["confidence"].isin(["high","medium"]))]
        added = False
        for t,v in scope_filters:
            sl = SMAP.get(t,t)
            for _,r in cat[(cat["filter"]==sl)&(cat["filter_value"]==v)].head(1).iterrows():
                items.append({"type":"Insight","text":r["text"],"scope":f"{sl} = {v}"}); added=True
        if not added:
            for _,r in cat[cat["filter"]=="Global"].head(1).iterrows():
                items.append({"type":"Insight","text":r["text"],"scope":_scope_txt("Global","")})
    return items[:max_items]


# ── statistical helpers ───────────────────────────────────────────────────────
def compute_uplift(sub, feat, mc):
    if feat not in sub.columns or mc not in sub.columns: return None,None,None
    if feat in BIN_FEATS:
        g1=sub.loc[sub[feat]==1,mc].dropna(); g0=sub.loc[sub[feat]==0,mc].dropna()
    else:
        mask=sub[feat].notna()&(sub[feat].astype(str).str.strip()!="")
        g1=sub.loc[mask,mc].dropna(); g0=sub.loc[~mask,mc].dropna()
    if len(g1)<3 or len(g0)<3: return None,None,None
    u=(g1.mean()-g0.mean())*100
    try: _,p=mannwhitneyu(g1,g0,alternative="two-sided")
    except: p=1.0
    sig="***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))
    return round(u,2),sig,len(g1)

def has_mask(sub, feat):
    if feat in BIN_FEATS: return sub[feat]==1
    return sub[feat].notna() & (sub[feat].astype(str).str.strip()!="")

def _sig(a, b):
    try: _,p=mannwhitneyu(a,b,alternative="two-sided")
    except: p=1.0
    return "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))

def feature_combinations(sub, target, mc, top_n=5):
    """Top binary feature partners for target feat on metric mc.
    Combined uplift = mean(both features) − mean(all assets in scope); significance = both vs the rest of scope."""
    if target not in sub.columns or mc not in sub.columns: return []
    gm = sub[mc].dropna().mean()*100
    mt = has_mask(sub, target)
    g1t = sub.loc[mt,mc].dropna()
    if len(g1t)<3: return []
    solo_t = g1t.mean()*100 - gm
    rows=[]
    for f in BIN_FEATS:
        if f==target or f not in sub.columns: continue
        mb = mt & (sub[f]==1)
        nb=int(mb.sum())
        if nb<3: continue
        g_both = sub.loc[mb,mc].dropna(); g_rest = sub.loc[~mb,mc].dropna()
        cu=g_both.mean()*100-gm
        g1p=sub.loc[sub[f]==1,mc].dropna()
        sp=g1p.mean()*100-gm if len(g1p)>=3 else 0
        best_single = min(solo_t, sp) if lower_better(mc) else max(solo_t, sp)
        sig = _sig(g_both, g_rest) if len(g_rest)>=3 else "ns"
        rows.append({"partner":f,"label":FEAT_LABEL.get(f,f),
                     "combined":round(cu,2),"solo_t":round(solo_t,2),
                     "solo_p":round(sp,2),"synergy":round(cu-best_single,2),"n":nb,
                     "n_rest":len(g_rest),"sig":sig,"coverage":nb/max(len(sub),1)})
    rows.sort(key=lambda x:x["combined"],reverse=not lower_better(mc))
    return rows[:top_n]

def value_uplifts(sub, feat, mc, min_n=5):
    """Uplift per individual value of a categorical feature.
    Reference = assets with no value for the feature; for always-filled features, all other assets.
    Values are matched as whole items of comma-separated lists (so 'low' never matches 'very_low')."""
    if feat not in sub.columns or mc not in sub.columns or feat in BIN_FEATS: return []
    s = sub[feat].astype(str).where(sub[feat].notna(), "")
    d = s.str.get_dummies(sep=",")
    if d.empty: return []
    d.columns = d.columns.str.strip()
    d = d.loc[:, [c for c in d.columns if c and c != "nan"]]
    d = d.T.groupby(level=0).max().T
    present = d.sum(axis=1) > 0
    absent = sub.loc[~present, mc].dropna()
    rows = []
    for val in d.columns:
        m = d[val] == 1
        if m.sum() < min_n: continue
        g = sub.loc[m, mc].dropna()
        use_absent = len(absent) >= min_n
        ref = absent if use_absent else sub.loc[~m, mc].dropna()
        if len(g) < min_n or len(ref) < min_n: continue
        rows.append({"value": val, "uplift": round((g.mean()-ref.mean())*100, 2), "sig": _sig(g, ref),
                     "n": len(g), "n_ref": len(ref),
                     "ref": "assets without any value" if use_absent else "all other assets"})
    rows.sort(key=lambda x: abs(x["uplift"]), reverse=True)
    return rows

STAB_DIMS = [("OU","operating_unit_code"),("Market","country_name"),("Format","asset_category")]
def stability(sub, mask, mc, min_side=5):
    """Does a feature/combination (boolean mask over sub) hold across OUs, markets and formats?"""
    a = sub.loc[mask, mc].dropna(); b = sub.loc[~mask, mc].dropna()
    if len(a) < 3 or len(b) < 3: return pd.DataFrame(), pd.DataFrame()
    u0 = (a.mean()-b.mean())*100
    summ, det = [], []
    for dim, col in STAB_DIMS:
        if col not in sub.columns: continue
        segs = []
        for seg, g in sub.groupby(col):
            m = mask.loc[g.index]
            x = g.loc[m, mc].dropna(); y = g.loc[~m, mc].dropna()
            if len(x) < min_side or len(y) < min_side: continue
            u = (x.mean()-y.mean())*100
            segs.append(seg)
            det.append({"Dimension": dim, "Segment": seg, "Uplift (pp)": round(u,1),
                        "Sig.": _sig(x, y), "n with": len(x), "n without": len(y),
                        "Same direction as overall": "Yes" if np.sign(u)==np.sign(u0) else "No"})
        if len(segs) < 2:
            summ.append({"Dimension": dim, "Segments tested": len(segs), "Same direction": "—",
                         "Verdict": "Not enough segments"}); continue
        same = sum(1 for d in det if d["Dimension"]==dim and d["Same direction as overall"]=="Yes")
        driver = None
        for seg in segs:   # leave one segment out
            keep = sub[col] != seg
            x = sub.loc[keep & mask, mc].dropna(); y = sub.loc[keep & ~mask, mc].dropna()
            if len(x) < 3 or len(y) < 3: driver = seg; break
            u_lo = (x.mean()-y.mean())*100
            if np.sign(u_lo) != np.sign(u0) or abs(u_lo) < abs(u0)*.5: driver = seg; break
        share = same/len(segs)
        verdict = (f"Driven by {driver}" if driver is not None else
                   "Holds broadly" if share >= .75 else "Mixed" if share >= .5 else "Inconsistent")
        summ.append({"Dimension": dim, "Segments tested": len(segs),
                     "Same direction": f"{same} of {len(segs)}", "Verdict": verdict})
    return pd.DataFrame(summ), pd.DataFrame(det)

def render_stability(sub, mask, mc, what, key):
    summ, det = stability(sub, mask, mc)
    if summ.empty:
        st.markdown('<p style="color:#AAA;font-size:.8rem">Not enough assets for a stability check.</p>', unsafe_allow_html=True); return
    st.markdown(f'<div class="def-note">Is the effect of <strong>{what}</strong> on {M_LABEL.get(mc,mc)} consistent? '
                'Uplift is recomputed inside each OU, market and format (segments need 5+ assets with and without). '
                '<strong>Driven by X</strong> = removing that one segment flips or halves the overall result.</div>',
                unsafe_allow_html=True)
    st.dataframe(summ, hide_index=True, use_container_width=True)
    if not det.empty:
        with st.expander("Segment detail"):
            st.dataframe(det, hide_index=True, use_container_width=True)
            dl(det, f"stability_{what}", key)


# ── precomputed combo helpers ─────────────────────────────────────────────────
def get_scope_key(scope_filters):
    if not scope_filters: return "global||All"
    if len(scope_filters)==1:
        t,v=scope_filters[0]; k=f"{t}||{v}"
        return k if k in results else None
    return None

def get_combo(sub_df, scope_key, mc, min_n=5):
    if scope_key and scope_key in results and mc in results[scope_key]:
        c=results[scope_key][mc]
        return c["combo"],c["steps"],c["baseline_pp"],c["n_total"]
    return None,[],0,len(sub_df)

@st.cache_data(show_spinner=False)
def feature_drivers(idx_key, min_n):
    """Positive / negative feature drivers across the headline metrics for the selection.
    Binary features: with vs without. Categorical features: each value vs assets without any value."""
    sub = df_full.loc[list(idx_key)]
    pos, neg = [], []
    for feat in ALL_FEATS:
        if feat not in sub.columns: continue
        for mc in ["Attention_T2B","Persuasion_T2B","Likeability_Love_Like_T2B",
                   "Shareability_T2B","Experience_Recall_T2B","Tiredness_T2B"]:
            if mc not in sub.columns: continue
            mlbl, lbl0 = M_LABEL.get(mc, mc), FEAT_LABEL.get(feat, feat)
            if feat in BIN_FEATS:
                g1 = sub.loc[sub[feat]==1, mc].dropna(); g0 = sub.loc[sub[feat]==0, mc].dropna()
                if len(g1) < 2 or len(g0) < 2: continue
                cand = [(lbl0, (g1.mean()-g0.mean())*100, _sig(g1,g0), len(g1), len(g0))]
            else:
                cand = [(f"{lbl0}: {r['value']}", r["uplift"], r["sig"], r["n"], r["n_ref"])
                        for r in value_uplifts(sub, feat, mc, min_n=min_n)]
            for lbl, u, sg, n, n0 in cand:
                (pos if is_good(u, mc) else neg).append((lbl, mlbl, round(u,2), sg, n, rel_label(sg, n, n0)))
    return pos, neg

def live_min_n(n_scope):
    return 30 if n_scope >= 1000 else (15 if n_scope >= 150 else 5)

@st.cache_data(show_spinner=False)
def live_combos(idx_key, mc, min_n, n_combos=3, depth=3):
    """Greedy best combinations computed on the current selection.
    Starts from the mean of all assets in scope and adds, one at a time, the feature value that moves the
    metric most in the good direction (up; down for Tiredness). Alternatives exclude features already used."""
    sub = df_full.loc[list(idx_key)]
    if mc not in sub.columns: return []
    y = sub[mc].to_numpy(dtype=float); ok = ~np.isnan(y)
    if ok.sum() < max(min_n, 3): return []
    sign = -1 if lower_better(mc) else 1
    base = y[ok].mean()*100
    cands = []
    for f in ALL_FEATS:
        if f not in sub.columns: continue
        lbl = FEAT_LABEL.get(f, f)
        if f in BIN_FEATS:
            for v in (1, 0):
                m = (sub[f]==v).to_numpy()
                if (m & ok).sum() >= min_n: cands.append((f, v, m, lbl if v==1 else f"No {lbl}"))
        else:
            col = sub[f].astype(str).where(sub[f].notna(), "")
            for v, c in col.value_counts().items():
                if v and v != "nan" and c >= min_n: cands.append((f, v, (col==v).to_numpy(), f"{lbl}: {v}"))
    out, used = [], set()
    for rank in range(1, n_combos+1):
        cur, cur_mean, steps, combo, fset = ok.copy(), base, [], [], set()
        for step in range(1, depth+1):
            best = None
            for f, v, m, lbl in cands:
                if f in fset or f in used: continue
                mm = cur & m; n = int(mm.sum())
                if n < min_n: continue
                mean = y[mm].mean()*100; gain = (mean-cur_mean)*sign
                if gain > 0 and (best is None or gain > best[0]): best = (gain, f, v, mm, lbl, mean, n)
            if best is None: break
            _, f, v, mm, lbl, mean, n = best
            steps.append({"step":step, "feat":f, "label":lbl, "step_gain_pp":round(mean-cur_mean,2),
                          "cumulative_uplift_pp":round(mean-base,2), "metric_mean":round(mean,2), "n":n})
            combo.append((f, v)); fset.add(f); cur, cur_mean = mm, mean
        if not steps: break
        used |= fset
        out.append({"rank":rank, "combo":combo, "steps":steps, "baseline_pp":round(base,2), "n_total":int(ok.sum())})
    return out

def get_top_combos(sub_df, scope_key, mc):
    """Precomputed combinations when this exact scope was precomputed; otherwise computed live
    (year filter, campaign, several filters at once) — and always live for Tiredness (lower is better)."""
    if scope_key and scope_key in results and mc in results[scope_key] and not lower_better(mc):
        c = results[scope_key][mc]
        top3 = c.get("top3_combos", [])
        if not top3 and c.get("steps"):
            top3 = [{"rank":1,"combo":c["combo"],"steps":c["steps"],
                     "baseline_pp":c["baseline_pp"],"n_total":c["n_total"]}]
        if top3: return top3, "precomputed"
    return live_combos(tuple(sub_df.index), mc, live_min_n(len(sub_df))), "live"

def default_sel(combo):
    sel={f:"__any__" for f in ALL_FEATS}
    if combo:
        for feat,val in combo:
            if feat in ALL_FEATS:
                sel[feat]="1" if val==1 else ("0" if val==0 else str(val))
    return sel

def apply_sel(sub_df,sel):
    mask=pd.Series(True,index=sub_df.index)
    for feat,val in sel.items():
        if val=="__any__" or feat not in sub_df.columns: continue
        mask&=(sub_df[feat]==(1 if val=="1" else 0)) if feat in BIN_FEATS else (sub_df[feat]==val)
    return mask

def score_sel(sub_df,sel):
    mask=apply_sel(sub_df,sel); n=int(mask.sum())
    return {mc:(round(sub_df.loc[mask,mc].dropna().mean()*100,2),n) if n>0 else (None,0)
            for mc in ALL_METRICS if mc in sub_df.columns}

def get_opts(feat,scope_key,mc,sub_df):
    fv=None
    if scope_key and scope_key in results and mc in results[scope_key]:
        fv=results[scope_key][mc].get("feature_value_uplift",{}).get(feat) or \
           results[scope_key][mc].get("feature_options",{}).get(feat)
    raws,disp=["__any__"],["Any"]
    if fv:
        for row in fv:
            if row.get("n",0)<3: continue
            raws.append(str(row["val"])); disp.append(str(row["label"]))
    elif feat in BIN_FEATS and feat in sub_df.columns:
        if int((sub_df[feat]==1).sum())>=3: raws+=["1"]; disp+=["Yes"]
        if int((sub_df[feat]==0).sum())>=3: raws+=["0"]; disp+=["No"]
    elif feat in sub_df.columns:
        for v in sorted(sub_df[feat].dropna().astype(str).unique()):
            if int((sub_df[feat].astype(str)==v).sum())>=3: raws.append(v); disp.append(v[:32])
    return raws,disp

def short_list(txt, k=3):
    items = [x.strip() for x in str(txt).split(",") if x.strip()]
    return ", ".join(items[:k]) + (f" +{len(items)-k} more" if len(items) > k else "")


def make_heatmap(piv,title,fs=(13,5)):
    cmap=LinearSegmentedColormap.from_list("rg",["#D32F2F","white","#2E7D32"],N=256)
    vals=piv.values.astype(float); vmax=max(np.nanpercentile(np.abs(vals),95),1.0)
    fig,ax=plt.subplots(figsize=fs); fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
    im=ax.imshow(vals,cmap=cmap,vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,fontsize=8.5,rotation=30,ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=8.5)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v=vals[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=7.5,
                        color="white" if abs(v)>vmax*.55 else DARK)
            else:
                ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fc="#E8E4DE",ec="white",lw=.5))
    plt.colorbar(im,ax=ax,shrink=.55,label="Uplift (pp)")
    ax.set_title(title,fontsize=10,fontweight="bold",color=DARK,pad=8)
    ax.spines[:].set_visible(False); plt.tight_layout()
    return fig


# ── top bar ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">The Coca&#8209;Cola Company</div>
    <div class="topbar-pipe"></div>
    <div class="topbar-sub">Asset Intelligence</div>
  </div>
  <div class="topbar-sub">{len(df_full):,} assets &nbsp;·&nbsp; Asset Feature Explorer</div>
</div>
""", unsafe_allow_html=True)

# ── navigation (horizontal, top of page) ──────────────────────────────────────
PAGES = [
    "ℹ · About this App",
    "01 · Overview & Performance",
    "02 · Feature Impact",
    "03 · Combination Explorer",
    "04 · Feature Combinations & OU Impact",
    "05 · Insight Catalog",
    "06 · Rulebook",
    "07 · Campaign Comparison",
]
page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed", key="nav")

# ══════════════════════════════════════════════════════════════════════════════
# FILTER BAR — rendered per-page so hero appears above filters
# ══════════════════════════════════════════════════════════════════════════════
CTX = {}   # filled by render_filters: benchmark population, years, scope signature

def render_filters():
    """Render the filter bar and return (sub_df, scope_filters, scope_key, min_n, scope_label).
    Call this inside each page block, after the hero. Benchmark info goes into CTX."""
    st.markdown('<div class="sec-label" style="margin-top:1.4rem">Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1, .8, 1.4, 1.3])
    with fc1:
        ou_opts = ["All OUs"] + sorted(df_full["operating_unit_code"].dropna().unique().tolist())
        sel_ou = st.selectbox("Operating Unit", ou_opts, key="f_ou")
    with fc2:
        brand_opts = ["All Brands"] + sorted(df_full["brand_name"].dropna().unique().tolist())
        sel_brand = st.selectbox("Brand", brand_opts, key="f_brand")
    with fc3:
        year_opts = sorted(int(y) for y in df_full["_year"].dropna().unique())
        sel_years = st.multiselect("Year", year_opts, default=[], key="f_year") if year_opts else []
    with fc4:
        _csub = df_full.copy()
        if sel_ou != "All OUs": _csub = _csub[_csub["operating_unit_code"] == sel_ou]
        if sel_brand != "All Brands": _csub = _csub[_csub["brand_name"] == sel_brand]
        if sel_years: _csub = _csub[_csub["_year"].isin(sel_years)]
        _cdf = (_csub[["campaign_sk_id","campaign_display_name","campaign_code"]]
                .drop_duplicates("campaign_sk_id").sort_values("campaign_display_name"))
        camp_opts = ["All Campaigns"] + [f"{r.campaign_display_name} ({r.campaign_code})"
                                          for r in _cdf.itertuples()]
        camp_ids  = [None] + _cdf["campaign_sk_id"].tolist()
        sel_camp_i = st.selectbox("Campaign", range(len(camp_opts)),
                                   format_func=lambda i: camp_opts[i], key="f_camp")
        sel_camp_i = min(sel_camp_i, len(camp_ids)-1)
        sel_camp = camp_ids[sel_camp_i]
    with fc5:
        mkt_opts = sorted(df_full["country_name"].dropna().unique().tolist())
        sel_mkts = st.multiselect("Market(s)", mkt_opts, default=[], key="f_mkts")

    # apply filters
    sub_df = df_full.copy()
    scope_filters = []
    if sel_ou != "All OUs":
        sub_df = sub_df[sub_df["operating_unit_code"] == sel_ou]
        scope_filters.append(("ou", sel_ou))
    if sel_brand != "All Brands":
        sub_df = sub_df[sub_df["brand_name"] == sel_brand]
        scope_filters.append(("brand", sel_brand))
    if sel_years:
        sub_df = sub_df[sub_df["_year"].isin(sel_years)]
    if sel_camp is not None:
        sub_df = sub_df[sub_df["campaign_sk_id"] == sel_camp]
    if sel_mkts:
        sub_df = sub_df[sub_df["country_name"].isin(sel_mkts)]
        for m in sel_mkts:
            scope_filters.append(("market", m))

    # comparison benchmark ("vs average")
    bench_base = df_full[df_full["_year"].isin(sel_years)] if sel_years else df_full
    yr_txt = f" in {', '.join(map(str, sel_years))}" if sel_years else ""
    bench_opts = ["All assets (all brands)"]
    cats = sorted(sub_df["category"].dropna().unique().tolist()) if "category" in sub_df.columns else []
    if cats and len(cats) <= 3: bench_opts.append("Same category")
    if "_tccc" in df_full.columns: bench_opts.append("TCCC brands only")
    b1, b2 = st.columns([1.3, 3.7])
    with b1:
        bench_choice = st.selectbox("Compare against", bench_opts, key="f_bench")
    if bench_choice == "Same category":
        bench_df = bench_base[bench_base["category"].isin(cats)]
        bench_label = f"average of all {' / '.join(cats)} assets{yr_txt}"
    elif bench_choice == "TCCC brands only":
        bench_df = bench_base[bench_base["_tccc"]]
        bench_label = f"average of all TCCC-brand assets{yr_txt} (competitors excluded)"
    else:
        bench_df = bench_base
        bench_label = f"average of all assets in the dataset{yr_txt}, all brands incl. competitors"
    with b2:
        st.markdown(f'<div class="def-note" style="margin-top:1.55rem"><strong>“vs average”</strong> = your selection '
                    f'compared with the {bench_label} ({len(bench_df):,} assets). '
                    f'<strong>“vs scope”</strong> = compared with the {len(sub_df):,} assets matching your filters. '
                    f'All averages are simple means over assets — every asset counts once, no weighting by market.</div>',
                    unsafe_allow_html=True)

    scope_key = get_scope_key(scope_filters[:1]) if (sel_camp is None and not sel_years) else None
    min_n = 5 if (scope_filters or sel_years) else 10

    chip_parts = []
    if sel_ou != "All OUs":    chip_parts.append(f"OU: {sel_ou}")
    if sel_brand != "All Brands": chip_parts.append(f"Brand: {sel_brand}")
    if sel_years:              chip_parts.append(f"Year: {', '.join(map(str, sel_years))}")
    if sel_camp is not None:   chip_parts.append(f"Campaign: {camp_opts[sel_camp_i].split('(')[0].strip()}")
    if sel_mkts:               chip_parts.append(f"Markets: {', '.join(sel_mkts[:2])}{'...' if len(sel_mkts)>2 else ''}")
    if chip_parts:
        chips = "".join(f'<span class="scope-chip">{p}</span>' for p in chip_parts)
        st.markdown(f'<div style="margin:.4rem 0 0 0">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:.74rem;color:#CCC8C2;margin:.4rem 0 0 0">All brands · All years · All campaigns · All markets</p>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    scope_label = "Global" if not chip_parts else " · ".join(chip_parts)
    CTX.update(bench_df=bench_df, bench_label=bench_label, years=sel_years,
               scope_sig=f"{scope_filters}|{sel_years}|{sel_camp}", n_scope=len(sub_df))
    return sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 00 — ABOUT THIS APP (no filters needed)
# ══════════════════════════════════════════════════════════════════════════════
if page == "ℹ · About this App":
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">About this App</div>
      <div class="hero-title">Asset Intelligence</div>
      <div class="hero-sub">A tool for Coca-Cola data analysts to understand which creative
        features drive ad performance — and where, and why. Built on survey data and
        statistical uplift analysis across thousands of tested assets.</div>
    </div>""", unsafe_allow_html=True)

    # ── Best practice ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Best practice — what to always note</div>', unsafe_allow_html=True)
    st.markdown("""<div class="takeaway"><div class="takeaway-title">Before you quote any number</div><ul>
      <li><strong>Scope and comparison.</strong> Write down your filters and the <em>Compare against</em> setting —
        “+2pp vs average” means nothing without saying which average.</li>
      <li><strong>Reliability.</strong> Only brief from <strong>High</strong> findings; use <strong>Medium</strong> as
        direction; treat <strong>Directional</strong> as a hypothesis to test.</li>
      <li><strong>N and coverage.</strong> Note how many assets support the result and what share of your scope they are.
        Under 10 assets is anecdotal.</li>
      <li><strong>Stability.</strong> Check the stability view: a result “driven by” one OU, market or format does not
        generalise.</li>
      <li><strong>Alerts.</strong> Read every high-severity alert for your feature — Heterogeneity (reversal in a scope)
        and Conflict (trade-off between metrics) can overturn a finding.</li>
      <li><strong>Tiredness is reversed.</strong> Higher Tiredness is worse; a positive uplift on Tiredness is bad news.</li>
    </ul></div>
    <div class="takeaway" style="border-left-color:#2A8050"><div class="takeaway-title" style="color:#2A8050">The takeaways that matter most</div><ul>
      <li><strong>01 Overview:</strong> is your selection above or below its benchmark — and its local market average?</li>
      <li><strong>02 Feature Impact:</strong> the 2–3 High-reliability features that move your priority metric.</li>
      <li><strong>03 Combination Explorer:</strong> the best combination, its n, and whether it holds across segments.</li>
      <li><strong>04 OU Impact:</strong> any OU where your chosen feature reverses.</li>
      <li><strong>05/06 Catalog &amp; Rulebook:</strong> the Key takeaways box and high-severity alerts for your scope.</li>
      <li><strong>07 Campaign Comparison:</strong> campaign score vs local market average, for campaign reports.</li>
    </ul></div>""", unsafe_allow_html=True)

    # ── How to use ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">How to use this app</div>', unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#fff;border:1px solid #EAE8E2;border-radius:8px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
      <div style="font-size:.88rem;color:#333;line-height:1.8;">
        The app is structured to answer three questions in order:
      </div>
      <div style="display:flex;gap:1rem;margin-top:.9rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:180px;background:#EBF3FB;border-radius:6px;padding:.8rem 1rem;">
          <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#2D5BE3;margin-bottom:.3rem">1 · What happened?</div>
          <div style="font-size:.83rem;color:#1C1C1C;font-weight:600">Overview &amp; Performance</div>
          <div style="font-size:.78rem;color:#555;margin-top:.2rem">How did the campaign perform across all metrics? Which assets stand out?</div>
        </div>
        <div style="flex:1;min-width:180px;background:#EAFAF1;border-radius:6px;padding:.8rem 1rem;">
          <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#2A8050;margin-bottom:.3rem">2 · Why?</div>
          <div style="font-size:.83rem;color:#1C1C1C;font-weight:600">Feature Impact &amp; Combinations</div>
          <div style="font-size:.78rem;color:#555;margin-top:.2rem">Which creative features drove those results? What combinations work best?</div>
        </div>
        <div style="flex:1;min-width:180px;background:#F3EEFF;border-radius:6px;padding:.8rem 1rem;">
          <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#7030A0;margin-bottom:.3rem">3 · So what?</div>
          <div style="font-size:.83rem;color:#1C1C1C;font-weight:600">Catalog, Rulebook &amp; Market Impact</div>
          <div style="font-size:.78rem;color:#555;margin-top:.2rem">What should change in the next brief? Where do findings not apply?</div>
        </div>
      </div>
      <div style="font-size:.8rem;color:#888;margin-top:.9rem;padding-top:.7rem;border-top:1px solid #F0EEE8;">
        Use the <strong>Filters</strong> at the top to scope results by OU, Brand, Year, Campaign, or Market, and
        <strong>Compare against</strong> to choose the benchmark behind every “vs average”. Every table has a
        <strong>Download CSV</strong> button.
        All analysis updates in real time. Start with <strong>01 · Overview &amp; Performance</strong>
        and work through the pages in order.
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Page guide ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Page guide</div>', unsafe_allow_html=True)

    pages = [
        ("01 · Overview & Performance", "what", "WHAT HAPPENED",
         "Your starting point. Shows all 8 metric KPI scores for the current filter, "
         "a data-driven insight summary (Shareability, Recall, Tiredness highlighted), "
         "campaign performance breakdown, and the top-performing assets ranked by any metric.",
         "Start here after applying your filters. The insight summary at the top saves you "
         "from having to manually interpret every number — it surfaces what matters."),
        ("02 · Feature Impact", "why", "WHY",
         "For every creative feature, shows how much higher or lower a chosen metric scores "
         "when that feature is present vs absent. Sorted by effect size with significance stars. "
         "Includes a per-value breakdown for Occasions and Passion Points.",
         "Use this to answer: <em>which specific creative choices made the difference?</em> "
         "Only act on *** or ** findings. ns = not significant → do not brief from it."),
        ("03 · Combination Explorer", "why", "WHY",
         "Shows the top greedy feature combinations per metric — the algorithm picks the "
         "sequence of features that, together, maximise the score. Up to 3 distinct combinations "
         "per metric. The Explorer lets you test any custom combination in real time.",
         "Use this to build a multi-feature brief. The scoreboard shows trade-offs across "
         "all 8 metrics simultaneously so you can spot if a combination hurts a metric "
         "while helping another."),
        ("04 · Feature Combinations & OU Impact", "why", "WHY",
         "Two-part page: (A) synergy analysis — which features amplify each other's effect, "
         "and (B) OU breakdown — how the same feature performs differently across Operating Units. "
         "Starts with a “So what” summary. Red bars in the OU chart = the feature reverses in that OU.",
         "Critical step before briefing. A feature that works at +5pp globally "
         "may be −3pp in a specific OU. Always check the OU view before writing a global brief."),
        ("05 · Insight Catalog", "sowhat", "SO WHAT",
         "The validated evidence base: every statistically tested finding from the full pipeline, "
         "filterable by scope, metric, reliability, and effect. Starts with Key takeaways; includes a market impact "
         "section to see if findings hold across markets.",
         "High-reliability findings are safe to use in briefs directly. "
         "Medium = directional only. Directional = do not brief on its own. "
         "Filter by OU or Brand to see scope-specific findings."),
        ("06 · Rulebook", "sowhat", "SO WHAT",
         "Automated alerts: Conflicts (feature improves one metric, hurts another), "
         "Heterogeneity (works on average but reverses in a specific scope), "
         "Opportunities (positive feature used in <20% of assets), "
         "Anti-patterns (negative across all metrics in a scope).",
         "Always review <strong>high-severity</strong> entries before writing any brief. "
         "Heterogeneity alerts are the most important — they prevent you from "
         "briefing something globally that will hurt specific markets."),
        ("07 · Campaign Comparison", "sowhat", "SO WHAT",
         "One row per campaign: markets, assets, average score, market-balanced score and the score "
         "versus each market's own average — so campaigns running in different markets can be compared.",
         "Use <em>vs local market avg</em> in campaign reports: it removes the effect of where a campaign ran."),
    ]

    for page_name, phase_key, phase_lbl, desc, tip in pages:
        phase_cls = f"phase-{phase_key}"
        st.markdown(f"""<div class="page-card">
          <div class="page-card-body">
            <div><span class="page-card-phase {phase_cls}">{phase_lbl}</span></div>
            <div class="page-card-title">{page_name}</div>
            <div class="page-card-desc">{desc}</div>
            <div class="page-card-tip"> <strong style="color:#1C1C1C;font-weight:700">Analyst tip:</strong> {tip}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Metrics reference ─────────────────────────────────────────────────────
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Metrics reference</div>', unsafe_allow_html=True)

    metric_defs = [
        ("Attention_T2B",             "Attention",          "Did the ad grab people's attention? Higher is better."),
        ("Persuasion_T2B",            "Persuasion",         "Did the ad make people more likely to buy? Higher is better."),
        ("Likeability_Love_Like_T2B", "Likeability",        "Did people love or like the ad? Higher is better."),
        ("Experience_Recall_T2B",     "Experience Recall",  "Do people remember seeing this ad? Higher = stronger brand salience."),
        ("Brand_Linkage_T2B",         "Brand Linkage",      "Do people correctly link the ad to the brand? Critical for attribution."),
        ("Uniqueness_T2B",            "Uniqueness",         "Does the ad feel distinctive vs competitors? Higher = more differentiated."),
        ("Shareability_T2B",          "Shareability",       "Would people share this ad? Proxy for organic reach and virality."),
        ("Tiredness_T2B",             "Tiredness",          "Are people tired of seeing this ad? Asked on a 5-point scale (5 = most tired, 1 = least tired); "
                                                            "the score is the % answering 4–5. ⚠ Lower is better — a positive uplift on Tiredness is bad."),
    ]

    col_m1, col_m2 = st.columns(2)
    for i, (mc_r, lbl_r, desc_r) in enumerate(metric_defs):
        col_r = col_m1 if i % 2 == 0 else col_m2
        dot_col = M_COLOR.get(mc_r, DARK)
        with col_r:
            st.markdown(f"""<div class="metric-ref-row">
              <div class="metric-ref-dot" style="background:{dot_col}"></div>
              <div>
                <div class="metric-ref-name">{lbl_r}</div>
                <div class="metric-ref-desc">{desc_r}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Statistical glossary ──────────────────────────────────────────────────
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Statistical glossary</div>', unsafe_allow_html=True)

    glossary = [
        ("pp (percentage points)", "Scores are expressed as the % of survey respondents giving a positive answer. "
         "62pp = 62% of viewers responded positively."),
        ("Uplift", "Mean score of assets WITH a feature minus mean score WITHOUT it. "
         "+3pp uplift means ads with that feature score 3 points higher on average."),
        ("Significance (*** ** * ns)", "How confident we are the result is not random noise. "
         "*** = p<0.001 (very confident) · ** = p<0.01 · * = p<0.05 (directional) · "
         "ns = not significant → do not use in a brief."),
        ("Baseline", "For feature uplift: the average score of assets in your scope that do NOT have the feature. "
         "For combinations: the average score of all assets in your scope."),
        ("vs average", "Your selection compared with the benchmark chosen in <em>Compare against</em> "
         "(all assets, same category, or TCCC brands only — limited to the selected year if one is set)."),
        ("vs scope", "An asset or segment compared with the average of all assets matching your filters."),
        ("vs local market average", "A market or campaign compared with the average of all benchmark assets in the "
         "same market — removes differences between markets' baselines (e.g. Fanta Nigeria vs all Nigeria assets)."),
        ("Reliability (High/Medium/Directional)", "High: significant at ** or *** with 50+ assets on each side. "
         "Medium: significant with 20+ on each side. Directional: not significant or small sample."),
        ("Coverage", "Share of assets in your scope that have the feature or combination."),
        ("Synergy", "The extra gain from combining two features beyond what each delivers alone. "
         "Positive synergy = the features amplify each other."),
    ]

    for term, defn in glossary:
        st.markdown(f"""<div style="padding:.55rem 0;border-bottom:1px solid #F5F3EF;display:flex;gap:.8rem;">
          <div style="font-size:.83rem;font-weight:700;color:#1C1C1C;width:200px;flex-shrink:0">{term}</div>
          <div style="font-size:.82rem;color:#555;line-height:1.6">{defn}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 01 — OVERVIEW & PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "01 · Overview & Performance":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">What Happened · Page 1 of 7</div>
      <div class="hero-title">Overview &amp; Performance</div>
      <div class="hero-sub">Your starting point. Metric scorecards for all 8 KPIs,
        a data-driven insight summary, campaign breakdown, and top assets ranked by any metric.
        Use this page to understand <em>what the numbers say</em> before digging into causes.</div>
    </div>""", unsafe_allow_html=True)
    how_to("""<strong>Analyst workflow:</strong> Start with the Insight Summary to get the headline read.
      Then check the Campaign breakdown to see if performance differs across campaigns.
      Use the Asset Viewer to inspect individual assets that stand out.
      Once you know what happened, move to <strong>02 · Feature Impact</strong> to find out why.""")

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()
    bench_df, bench_label = CTX["bench_df"], CTX["bench_label"]


    # KPI cards — row 1: assets + core 3
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-val">{len(sub_df):,}</div>
          <div class="kpi-lbl">Assets</div>
          <div class="kpi-sub">{sub_df["campaign_sk_id"].nunique():,} campaigns · {sub_df["country_name"].nunique():,} markets</div>
        </div>""", unsafe_allow_html=True)
    def _kpi(col, ml, mc):
        v = sub_df[mc].dropna()
        if v.empty: return
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {RED}">
              <div class="kpi-val">{v.mean()*100:.1f}<span style="font-size:1.1rem">pp</span></div>
              <div class="kpi-lbl">{ml}</div>
              <div class="kpi-sub">Average of {len(v):,} assets{" · lower is better" if lower_better(mc) else ""}</div>
            </div>""", unsafe_allow_html=True)
    for ml,(mc,col) in zip(
        ["Attention","Persuasion","Likeability"],
        [("Attention_T2B",k2),("Persuasion_T2B",k3),("Likeability_Love_Like_T2B",k4)]
    ):
        if mc in sub_df.columns: _kpi(col, ml, mc)

    # KPI cards — row 2: additional metrics
    n1,n2,n3,n4,n5 = st.columns(5)
    for ml,mc,col in [
        ("Experience Recall","Experience_Recall_T2B",n1),
        ("Brand Linkage",    "Brand_Linkage_T2B",    n2),
        ("Uniqueness",       "Uniqueness_T2B",        n3),
        ("Shareability",     "Shareability_T2B",      n4),
        ("Tiredness",        "Tiredness_T2B",         n5),
    ]:
        if mc in sub_df.columns: _kpi(col, ml, mc)

    # Glossary
    with st.expander("What do these numbers mean?"):
        st.markdown(f"""
**pp (percentage points)** — scores are expressed as the percentage of survey respondents who gave a positive answer.
A score of 62pp means 62% of people who watched the ad responded positively to that question.

**Average of N assets** — a simple mean across every asset in your selection. Each asset counts once; there is
**no weighting by market**, so markets with more tested assets carry more weight. Use *07 · Campaign Comparison*
for market-balanced scores.

**vs average** — your selection compared with the {bench_label}. Change it with *Compare against* in the filters.

**Attention** — did the ad successfully grab people's attention? (Top 2 Box survey response)

**Persuasion** — did the ad make people more likely to buy the product? (Top 2 Box)

**Likeability** — did people like the ad — did they love it or like it? (Love + Like combined)

**Tiredness** — 5-point scale, 5 = most tired. The score is the % answering 4–5, so lower is better.

**Uplift** — the difference in average score between ads that have a feature and ads that don't, within your selection. An uplift of +3pp means ads with that feature score 3 percentage points higher on average.
        """)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── CHANGE 13: Insight summary ────────────────────────────────────────────
    st.markdown('<div class="sec-label">Insight summary</div>', unsafe_allow_html=True)
    with st.expander("View performance summary across all metrics", expanded=True):
        st.markdown(f'<div class="def-note">Each card compares your selection ({len(sub_df):,} assets) with the '
                    f'<strong>{bench_label}</strong> ({len(bench_df):,} assets). Green = better, red = worse '
                    f'(for Tiredness, lower is better).</div>', unsafe_allow_html=True)
        # Compute per-metric means
        metric_summary = {}
        for mc_s, ml_s in M_LABEL.items():
            if mc_s in sub_df.columns:
                v = sub_df[mc_s].dropna()
                metric_summary[ml_s] = round(v.mean()*100, 1) if len(v)>0 else None

        # Benchmark means for comparison
        global_means = {}
        for mc_s, ml_s in M_LABEL.items():
            if mc_s in bench_df.columns:
                v = bench_df[mc_s].dropna()
                global_means[ml_s] = round(v.mean()*100, 1) if len(v)>0 else None

        # Build metric performance rows
        perf_html = '<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1rem">'
        sum_rows = []
        for ml_s, val_s in metric_summary.items():
            if val_s is None: continue
            gv = global_means.get(ml_s)
            diff = round(val_s - gv, 1) if gv else 0
            mc_s = mc_of(ml_s)
            diff_col = ucol(diff, mc_s, .5)
            sign = "+" if diff >= 0 else ""
            border_col = (GREEN if is_good(diff, mc_s) else RED) if abs(diff) > .5 else BORDER
            perf_html += (
                f'<div style="background:#fff;border:1px solid {border_col};border-radius:6px;padding:.55rem .9rem;min-width:120px">' +
                f'<div style="font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#888">{ml_s}{" ↓ better" if lower_better(mc_s) else ""}</div>' +
                f'<div style="font-size:1.35rem;font-weight:700;color:{DARK};font-family:Merriweather,serif">{val_s:.1f}pp</div>' +
                f'<div style="font-size:.72rem;color:{diff_col}">{sign}{diff:.1f}pp vs average</div>' +
                f'</div>'
            )
            sum_rows.append({"Metric": ml_s, "Selection (pp)": val_s, "Benchmark (pp)": gv,
                             "Difference (pp)": diff, "Benchmark": bench_label})
        perf_html += '</div>'
        st.markdown(perf_html, unsafe_allow_html=True)
        dl(pd.DataFrame(sum_rows), "metric_summary", "ov_sum")

        # Narrative
        shr_v = metric_summary.get("Shareability"); shr_g = global_means.get("Shareability")
        rec_v = metric_summary.get("Experience Recall"); rec_g = global_means.get("Experience Recall")
        trd_v = metric_summary.get("Tiredness"); trd_g = global_means.get("Tiredness")
        att_v = metric_summary.get("Attention"); att_g = global_means.get("Attention")

        narratives = []
        if att_v and att_g:
            diff_att = att_v - att_g
            narratives.append(f"**Attention** is {'above' if diff_att>0.5 else 'below' if diff_att<-0.5 else 'in line with'} the average at **{att_v:.1f}pp** ({'+' if diff_att>=0 else ''}{diff_att:.1f}pp).")
        if shr_v and shr_g:
            diff_shr = shr_v - shr_g
            if abs(diff_shr) > 0.5:
                narratives.append(f"**Shareability** is {'stronger' if diff_shr>0 else 'weaker'} than average at **{shr_v:.1f}pp** ({'+' if diff_shr>=0 else ''}{diff_shr:.1f}pp) — {'content has higher viral/word-of-mouth potential.' if diff_shr>0 else 'consider features that drive sharing behaviour.'}")
        if rec_v and rec_g:
            diff_rec = rec_v - rec_g
            if abs(diff_rec) > 0.5:
                narratives.append(f"**Experience Recall** is {'above' if diff_rec>0 else 'below'} average at **{rec_v:.1f}pp** ({'+' if diff_rec>=0 else ''}{diff_rec:.1f}pp) — {'strong brand salience.' if diff_rec>0 else 'ad may not be memorable enough — check brand cues and product presence.'}")
        if trd_v and trd_g:
            diff_trd = trd_v - trd_g
            if diff_trd > 0.5:
                narratives.append(f"⚠ **Tiredness** is above average at **{trd_v:.1f}pp** (+{diff_trd:.1f}pp) — worse than average (higher = more fatigue). Consider rotating creative or reducing frequency.")
            elif diff_trd < -0.5:
                narratives.append(f"✓ **Tiredness** is below average at **{trd_v:.1f}pp** ({diff_trd:.1f}pp) — better than average, audience is not fatigued.")

        # Top feature drivers (Attention)
        top_feat_rows = []
        for feat_s in ALL_FEATS:
            if feat_s not in sub_df.columns: continue
            u_s, sig_s, n_s = compute_uplift(sub_df, feat_s, "Attention_T2B")
            if u_s is not None and sig_s in ("***","**","*"):
                top_feat_rows.append((FEAT_LABEL.get(feat_s,feat_s), u_s, sig_s))
        top_feat_rows.sort(key=lambda x: abs(x[1]), reverse=True)
        if top_feat_rows:
            top_str = ", ".join(f"{n} ({u:+.1f}pp, {s})" for n,u,s in top_feat_rows[:3])
            narratives.append(f"**Top feature drivers (Attention):** {top_str}.")

        # Market nuances
        if "country_name" in sub_df.columns and sub_df["country_name"].nunique() > 1:
            mkt_att = (sub_df.groupby("country_name")["Attention_T2B"].mean().dropna()*100).sort_values(ascending=False)
            if len(mkt_att) >= 2:
                top_mkt = mkt_att.index[0]; bot_mkt = mkt_att.index[-1]
                diff_mkt = mkt_att.iloc[0] - mkt_att.iloc[-1]
                if diff_mkt > 3:
                    narratives.append(f"**Market nuances:** Attention varies significantly — **{top_mkt}** leads at {mkt_att.iloc[0]:.1f}pp vs **{bot_mkt}** at {mkt_att.iloc[-1]:.1f}pp ({diff_mkt:.1f}pp gap). Check feature performance by market before briefing.")

        for n_text in narratives:
            st.markdown(f'<div style="font-size:.85rem;color:#333;line-height:1.65;margin-bottom:.4rem">{n_text}</div>', unsafe_allow_html=True)

    # ── CHANGE 7: Campaign breakdown ──────────────────────────────────────────
    if sub_df["campaign_sk_id"].nunique() > 1:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Performance by campaign</div>', unsafe_allow_html=True)
        with st.expander(f"View {sub_df['campaign_sk_id'].nunique()} campaigns in this selection", expanded=False):
            camp_mc_sel = st.selectbox("Metric", METRIC_NAMES, key="camp_mc")
            camp_mc_col = mc_of(camp_mc_sel)
            tired_note(camp_mc_col)
            camp_grp = (sub_df.groupby(["campaign_sk_id","campaign_display_name"])
                        .agg(mean_val=(camp_mc_col, "mean"), n=(camp_mc_col, "count"),
                             countries=("country_name", lambda x: ", ".join(sorted(x.dropna().unique()))))
                        .reset_index())
            camp_grp["mean_val"] = camp_grp["mean_val"] * 100
            camp_grp = camp_grp.sort_values("mean_val", ascending=lower_better(camp_mc_col))
            global_mean_camp = bench_df[camp_mc_col].dropna().mean()*100 if camp_mc_col in bench_df.columns else None
            st.markdown(f'<div class="def-note">Score = average of the campaign\'s assets in your selection. '
                        f'vs Average = compared with the {bench_label}. For a comparison that adjusts for '
                        f'each market\'s own level, use <strong>07 · Campaign Comparison</strong>.</div>', unsafe_allow_html=True)
            tbl_camp = '<table style="width:100%;border-collapse:collapse;font-size:.82rem"><thead><tr style="border-bottom:2px solid #EAE8E2"><th style="text-align:left;padding:.4rem .6rem;color:#888">Campaign</th><th style="text-align:left;padding:.4rem .6rem;color:#888">Market(s)</th><th style="text-align:center;color:#888">Score</th><th style="text-align:center;color:#888">vs Average</th><th style="text-align:center;color:#888">n assets</th></tr></thead><tbody>'
            for i, (_, rc) in enumerate(camp_grp.iterrows()):
                bg_c = "#FAFAF8" if i%2==0 else "#FFF"
                diff_c = rc["mean_val"] - global_mean_camp if global_mean_camp else 0
                dc = ucol(diff_c, camp_mc_col, .5)
                sign_c = "+" if diff_c >= 0 else ""
                tbl_camp += (f'<tr style="background:{bg_c};border-bottom:1px solid #F5F3EF">'
                             f'<td style="padding:.35rem .6rem">{str(rc["campaign_display_name"])[:50]}</td>'
                             f'<td style="padding:.35rem .6rem;color:#555">{short_list(rc["countries"], 4)}</td>'
                             f'<td style="text-align:center;font-weight:600">{rc["mean_val"]:.1f}pp</td>'
                             f'<td style="text-align:center;color:{dc}">{sign_c}{diff_c:.1f}pp</td>'
                             f'<td style="text-align:center;color:#AAA">{int(rc["n"])}</td></tr>')
            tbl_camp += '</tbody></table>'
            st.markdown(tbl_camp, unsafe_allow_html=True)
            dl(camp_grp.assign(vs_average=(camp_grp["mean_val"]-global_mean_camp) if global_mean_camp else None)
                       .rename(columns={"campaign_display_name":"Campaign","countries":"Markets",
                                        "mean_val":f"{camp_mc_sel} (pp)","n":"n assets","vs_average":"vs average (pp)"}),
               "campaign_performance", "ov_camp")
            # Top features per campaign (top 3 campaigns)
            st.markdown('<div style="margin-top:1rem;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Top feature per campaign (largest uplift)</div>', unsafe_allow_html=True)
            for _, rc in camp_grp.head(3).iterrows():
                camp_sub = sub_df[sub_df["campaign_sk_id"]==rc["campaign_sk_id"]]
                if len(camp_sub) < 5: continue
                best_feat, best_u, best_sig = None, 0, "ns"
                for feat_c in BIN_FEATS:
                    if feat_c not in camp_sub.columns: continue
                    u_c,sig_c,_ = compute_uplift(camp_sub, feat_c, camp_mc_col)
                    if u_c is not None and abs(u_c) > abs(best_u):
                        best_feat, best_u, best_sig = feat_c, u_c, sig_c
                if best_feat:
                    col_c = ucol(best_u, camp_mc_col)
                    st.markdown(f'<div style="font-size:.8rem;padding:.2rem 0;color:#333"><strong>{str(rc["campaign_display_name"])[:40]}</strong> ({short_list(rc["countries"])}): {FEAT_LABEL.get(best_feat,best_feat)} <span style="color:{col_c}">{"+" if best_u>=0 else ""}{best_u:.1f}pp</span> ({best_sig})</div>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── Performance findings summary ──────────────────────────────────────────
    st.markdown('<div class="sec-label">Performance findings</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Which creative features drove positive or negative performance, and how each market performed.
    Use this to quickly identify what to repeat and what to avoid in the next brief.</div>""",
    unsafe_allow_html=True)

    _fs_col1, _fs_col2 = st.columns(2)

    with _fs_col1:
        # Feature drivers summary
        st.markdown('<div style="font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6A6660;margin-bottom:.4rem">Feature drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="def-note">Uplift = average of assets <em>with</em> the feature minus assets '
                    '<em>without</em> it, both inside your selection. Categorical features are split into their '
                    'values (e.g. <em>Seasonal: Christmas</em>), each compared with assets that have no value for '
                    'that feature. Tiredness drivers are flipped: raising Tiredness counts as negative.</div>',
                    unsafe_allow_html=True)
        pos_feats, neg_feats = feature_drivers(tuple(sub_df.index), 2 if len(sub_df) < 20 else 5)

        def _dedup(rows):
            seen = {}
            for e in sorted(rows, key=lambda x: (x[3]=="ns", -abs(x[2]))):
                k = (e[0], e[1])
                if k not in seen: seen[k] = e
            return list(seen.values())[:8]

        pos_feats = _dedup(pos_feats)
        neg_feats = _dedup(neg_feats)

        _small_n = len(sub_df) < 20
        if _small_n:
            st.markdown('<div style="font-size:.75rem;color:#906820;margin-bottom:.4rem">⚠ Small selection — showing all computed uplifts including ns (not significant). Treat as directional only.</div>', unsafe_allow_html=True)

        def _drv_rows(rows, good):
            for _lbl, _mlbl, _u, _sig, _n, _rel in rows:
                _sig_col = "#7030A0" if _sig in ("***","**","*") else "#CCC"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:.22rem 0;'
                    f'border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                    f'<span style="color:#1C1C1C"><strong>{_lbl}</strong> on {_mlbl}</span>'
                    f'<span style="display:flex;gap:.4rem;align-items:center">'
                    f'<span style="color:{GREEN if good else RED};font-weight:600">{_u:+.1f}pp</span>'
                    f'<span style="color:{_sig_col};font-size:.72rem">{_sig}</span>'
                    f'<span style="color:#CCC;font-size:.72rem">n={_n}</span>{rel_badge(_rel)}'
                    f'</span></div>',
                    unsafe_allow_html=True)

        if pos_feats:
            st.markdown('<div style="font-size:.75rem;color:#2A8050;font-weight:600;margin-bottom:.3rem">▲ Positive drivers</div>', unsafe_allow_html=True)
            _drv_rows(pos_feats, True)
        else:
            st.markdown('<p style="font-size:.8rem;color:#AAA">No positive drivers found.</p>', unsafe_allow_html=True)

        if neg_feats:
            st.markdown('<div style="font-size:.75rem;color:#C00020;font-weight:600;margin:.7rem 0 .3rem 0">▼ Negative drivers</div>', unsafe_allow_html=True)
            _drv_rows(neg_feats, False)
        _drv_cols = ["Feature","Metric","Uplift (pp)","Sig.","n with feature","Reliability"]
        dl(pd.concat([pd.DataFrame(pos_feats, columns=_drv_cols).assign(Effect="positive"),
                      pd.DataFrame(neg_feats, columns=_drv_cols).assign(Effect="negative")]),
           "feature_drivers", "ov_drv")

    with _fs_col2:
        # Market performance summary
        st.markdown('<div style="font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6A6660;margin-bottom:.4rem">Market performance</div>', unsafe_allow_html=True)
        _ref_lbl = st.selectbox("Metric", METRIC_NAMES, key="ov_mkt_mc")
        _ref_mc = mc_of(_ref_lbl)
        if _ref_mc in sub_df.columns and "country_name" in sub_df.columns:
            _m = (sub_df.groupby("country_name")[_ref_mc].agg(["mean","count"]).dropna())
            _loc = bench_df.groupby("country_name")[_ref_mc].mean()
            _avg_mean = bench_df[_ref_mc].dropna().mean() * 100
            mk = pd.DataFrame({"Market": _m.index, "Score (pp)": _m["mean"].values*100, "n assets": _m["count"].values})
            mk["Local market avg (pp)"] = mk["Market"].map(_loc).values*100
            mk["vs local market (pp)"] = mk["Score (pp)"] - mk["Local market avg (pp)"]
            mk["vs average (pp)"] = mk["Score (pp)"] - _avg_mean
            _local = mk["vs local market (pp)"].abs().max() > .05   # differs only when selection ⊂ market
            _key = "vs local market (pp)" if _local else "vs average (pp)"
            mk = mk.sort_values(_key, ascending=lower_better(_ref_mc))
            tired_note(_ref_mc)
            if _local:
                st.markdown(f'<div class="def-note">Each market is compared with <strong>its own average</strong> '
                            f'(all benchmark assets in that market) — e.g. your selection in Nigeria vs all Nigeria assets. '
                            f'Grey = vs the overall {bench_label} ({_avg_mean:.1f}pp).</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="def-note">Your selection covers whole markets, so each market is compared with '
                            f'the {bench_label} ({_avg_mean:.1f}pp). Filter a brand or campaign to compare with each '
                            f'market\'s own average.</div>', unsafe_allow_html=True)
            _good = mk[mk[_key].apply(lambda v: is_good(v, _ref_mc))]
            _bad  = mk[~mk.index.isin(_good.index)]
            for _title, _rows, _c in [("▲ Better than comparison", _good, GREEN), ("▼ Worse than comparison", _bad, RED)]:
                if _rows.empty: continue
                st.markdown(f'<div style="font-size:.75rem;color:{_c};font-weight:600;margin:.5rem 0 .3rem 0">{_title}</div>', unsafe_allow_html=True)
                for _, r in _rows.head(8).iterrows():
                    _extra = (f'<span style="color:#AAA;font-size:.72rem">{r["vs average (pp)"]:+.1f} vs avg</span>'
                              if _local else "")
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:.22rem 0;'
                        f'border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                        f'<span style="color:#1C1C1C">{r["Market"]} <span style="color:#CCC;font-size:.72rem">n={int(r["n assets"])}</span></span>'
                        f'<span style="display:flex;gap:.45rem;align-items:center"><span style="color:{_c};font-weight:600">'
                        f'{r["Score (pp)"]:.1f}pp ({r[_key]:+.1f}pp)</span>{_extra}</span></div>',
                        unsafe_allow_html=True)
            dl(mk.round(2), "market_performance", "ov_mkt")
        else:
            st.markdown('<p style="font-size:.8rem;color:#AAA">No market data available for this selection.</p>', unsafe_allow_html=True)

    # Top assets by metric
    st.markdown('<div class="sec-label">Top performing assets</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Select a metric below to rank assets by that score. Click <strong>▶ View</strong> to see the asset.
    Features listed are the binary elements present in each asset.</div>""",
    unsafe_allow_html=True)

    if "top_asset_metric" not in st.session_state:
        st.session_state.top_asset_metric = "Attention_T2B"

    btn_cols = st.columns(4)
    metric_btns = [
        ("Attention",        "Attention_T2B",             BLUE),
        ("Persuasion",       "Persuasion_T2B",            RED),
        ("Likeability",      "Likeability_Love_Like_T2B", GREEN),
        ("Exp. Recall",      "Experience_Recall_T2B",     "#B45309"),
    ]
    for col, (lbl, mc_key, mc_col) in zip(btn_cols, metric_btns):
        with col:
            if st.button(lbl, key=f"top_btn_{mc_key}"):
                st.session_state.top_asset_metric = mc_key
                st.rerun()
    btn_cols2 = st.columns(4)
    metric_btns2 = [
        ("Brand Linkage","Brand_Linkage_T2B",  "#0E7490"),
        ("Uniqueness",   "Uniqueness_T2B",      "#6D28D9"),
        ("Shareability", "Shareability_T2B",    "#065F46"),
        ("Tiredness",    "Tiredness_T2B",        "#9F1239"),
    ]
    for col, (lbl, mc_key, mc_col) in zip(btn_cols2, metric_btns2):
        with col:
            if st.button(lbl, key=f"top_btn_{mc_key}"):
                st.session_state.top_asset_metric = mc_key
                st.rerun()
    all_metric_btns = metric_btns + metric_btns2

    sort_mc = st.session_state.top_asset_metric
    sort_label = {m[1]: m[0] for m in all_metric_btns}[sort_mc]
    top_assets = sub_df[sub_df[sort_mc].notna()].sort_values(sort_mc, ascending=lower_better(sort_mc)).head(8)
    if lower_better(sort_mc):
        st.markdown('<div class="def-note">Ranked by <strong>lowest</strong> Tiredness — the least tiring assets come first.</div>',
                    unsafe_allow_html=True)
    dl(top_assets[[c for c in ["asset_name","campaign_display_name","country_name","brand_name","asset_category"]
                   + [m for m in M_LABEL if m in top_assets.columns] if c in top_assets.columns]],
       "top_assets", "ov_top")

    for _,r in top_assets.iterrows():
        feats=[FEAT_LABEL.get(f,f) for f in BIN_FEATS if r.get(f)==1]
        url=r.get("asset_url",""); name=r.get("asset_name",f"Asset {r.get('asset_sk_id','')}")
        link=f'<a href="{url}" target="_blank" style="color:{RED};font-size:.72rem;font-weight:600;text-decoration:none">▶ View</a>' if url else ""
        att=f"{r.get('Attention_T2B',0)*100:.0f}" if pd.notna(r.get("Attention_T2B")) else "—"
        pers=f"{r.get('Persuasion_T2B',0)*100:.0f}" if pd.notna(r.get("Persuasion_T2B")) else "—"
        like=f"{r.get('Likeability_Love_Like_T2B',0)*100:.0f}" if pd.notna(r.get("Likeability_Love_Like_T2B")) else "—"
        rec=f"{r.get('Experience_Recall_T2B',0)*100:.0f}" if pd.notna(r.get("Experience_Recall_T2B")) else "—"
        lnk=f"{r.get('Brand_Linkage_T2B',0)*100:.0f}" if pd.notna(r.get("Brand_Linkage_T2B")) else "—"
        uniq=f"{r.get('Uniqueness_T2B',0)*100:.0f}" if pd.notna(r.get("Uniqueness_T2B")) else "—"
        shr=f"{r.get('Shareability_T2B',0)*100:.0f}" if pd.notna(r.get("Shareability_T2B")) else "—"
        trd=f"{r.get('Tiredness_T2B',0)*100:.0f}" if pd.notna(r.get("Tiredness_T2B")) else "—"
        att_s  = f"font-weight:700;color:{BLUE}"   if sort_mc=="Attention_T2B" else ""
        trd_s  = f"font-weight:700;color:#9F1239" if sort_mc=="Tiredness_T2B"             else ""
        pers_s = f"font-weight:700;color:{RED}"   if sort_mc=="Persuasion_T2B"             else ""
        like_s = f"font-weight:700;color:{GREEN}" if sort_mc=="Likeability_Love_Like_T2B"  else ""
        rec_s  = f"font-weight:700;color:#B45309" if sort_mc=="Experience_Recall_T2B"      else ""
        lnk_s  = f"font-weight:700;color:#0E7490" if sort_mc=="Brand_Linkage_T2B"          else ""
        uniq_s = f"font-weight:700;color:#6D28D9" if sort_mc=="Uniqueness_T2B"             else ""
        shr_s  = f"font-weight:700;color:#065F46" if sort_mc=="Shareability_T2B"           else ""
        st.markdown(f"""<div class="asset-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div class="asset-name">{name}</div>{link}
          </div>
          <div class="asset-meta">{r.get("campaign_display_name","")} &nbsp;·&nbsp; {r.get("brand_name","")} &nbsp;·&nbsp; {r.get("country_name","")} &nbsp;·&nbsp; {r.get("asset_category","")}</div>
          <div class="asset-meta" style="margin-top:.14rem">Features: {", ".join(feats) if feats else "—"}</div>
          <div style="margin-top:.4rem">
            <span class="asc asc-a" style="{att_s}">Att {att}pp</span>
            <span class="asc asc-p" style="{pers_s}">Pers {pers}pp</span>
            <span class="asc asc-l" style="{like_s}">Like {like}pp</span>
            <span class="asc asc-a" style="{rec_s}">Rec {rec}pp</span>
            <span class="asc asc-a" style="{lnk_s}">Link {lnk}pp</span>
            <span class="asc asc-a" style="{uniq_s}">Uniq {uniq}pp</span>
            <span class="asc asc-a" style="{shr_s}">Share {shr}pp</span>
            <span class="asc asc-a" style="{trd_s}">Tired {trd}pp</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── Asset viewer widget ───────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Asset viewer</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Select any asset from the current selection to see its full feature profile and scores.</div>""",
    unsafe_allow_html=True)

    av_df = sub_df[sub_df["asset_name"].notna()].copy()
    av_df = av_df.sort_values([c for c in ["campaign_display_name","country_name","asset_name"] if c in av_df.columns])
    _av_lbl = (av_df.get("campaign_display_name", pd.Series("", index=av_df.index)).fillna("").astype(str) + " · " +
               av_df.get("country_name", pd.Series("", index=av_df.index)).fillna("").astype(str) + " · " +
               av_df.get("asset_category", pd.Series("", index=av_df.index)).fillna("").astype(str))
    _dup = _av_lbl.groupby(_av_lbl).cumcount() + 1
    _cnt = _av_lbl.map(_av_lbl.value_counts())
    av_df["_label"] = np.where(_cnt > 1, _av_lbl + " #" + _dup.astype(str), _av_lbl)
    av_options = av_df.index.tolist()
    if av_options:
        av_idx = st.selectbox("Select asset (campaign · market · format)", av_options,
                              format_func=lambda i: av_df.at[i, "_label"], key="asset_viewer_sel")
        av_row = av_df.loc[av_idx]
        av_sel = av_row["asset_name"]

        av_col1, av_col2 = st.columns([1.2, 1])
        with av_col1:
            # asset info card
            av_url  = av_row.get("asset_url", "")
            av_link = f'<a href="{av_url}" target="_blank" style="display:inline-block;margin-top:.6rem;background:{RED};color:#fff;padding:.35rem .9rem;border-radius:4px;font-size:.82rem;font-weight:600;text-decoration:none">▶ View asset</a>' if av_url else ""
            av_feats_bin = [FEAT_LABEL.get(f,f) for f in BIN_FEATS if av_row.get(f)==1]
            # Extended features: cat_feats + rich context columns
            EXTENDED_FEAT_COLS = list(CAT_FEATS) + [
                "occasions","passion_point","moments","music_style","seasonal",
                "emotions","food","products","intrinsic_elements",
                "additional_elements_and_product_placement",
                "most_frequently_used_word_in_creative",
            ]
            seen = set()
            av_feats_ext = {}
            av_feats_ext_keys = {}  # label -> feat_key for description lookup
            for f in EXTENDED_FEAT_COLS:
                if f in seen: continue
                seen.add(f)
                val = av_row.get(f)
                if pd.notna(val) and str(val).strip() not in ("","nan"):
                    label = FEAT_LABEL.get(f, f.replace("_"," ").title())
                    av_feats_ext[label] = str(val)
                    av_feats_ext_keys[label] = f

            # Build objects HTML
            av_objects = str(av_row.get("objects","")).strip()
            obj_html_av = ""
            if av_objects and av_objects != "nan":
                obj_tags = "".join(
                    f'<span style="display:inline-block;background:#EBF3FB;border-radius:3px;padding:.1rem .45rem;margin:.1rem .15rem 0 0;font-size:.73rem;color:#2D5BE3">{o.strip()}</span>'
                    for o in av_objects.split(",") if o.strip()
                )
                obj_html_av = f'''<div style="margin-top:.8rem">
                  <div style="font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6A6660;margin-bottom:.3rem">Detected objects</div>
                  <div style="line-height:2">{obj_tags}</div></div>'''

            # Build feature rows with descriptions
            feat_rows_html = ""
            for lbl, val in av_feats_ext.items():
                fkey = av_feats_ext_keys.get(lbl, "")
                desc = FEAT_DESC.get(fkey, "")
                feat_rows_html += (
                    f'<div style="padding:.28rem 0;border-bottom:1px solid #F5F3EF">' +
                    f'<div style="display:flex;gap:.5rem"><span style="font-size:.78rem;color:#888;width:150px;flex-shrink:0">{lbl}</span><span style="font-size:.82rem;color:#1C1C1C;font-weight:500">{val}</span></div>'
                )
                if desc:
                    feat_rows_html += f'<div style="font-size:.71rem;color:#AAA;padding-left:155px;margin-top:.06rem">{desc}</div>'
                feat_rows_html += '</div>'
            if not feat_rows_html:
                feat_rows_html = '<div style="font-size:.83rem;color:#AAA">No feature data</div>'

            av_caption = str(av_row.get("asset_caption", "")).strip()
            caption_html = ""
            if av_caption and av_caption != "nan":
                caption_html = f'''<div style="margin-top:.8rem;padding:.6rem .8rem;background:#F5F3EF;border-radius:4px;">
                  <div style="font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6A6660;margin-bottom:.25rem">AI Caption</div>
                  <div style="font-size:.8rem;color:#444;line-height:1.55;font-style:italic">{av_caption[:300]}{"…" if len(av_caption)>300 else ""}</div>
                </div>'''
            st.markdown(f"""<div style="background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.2rem 1.4rem;">
              <div style="font-size:.95rem;font-weight:600;color:#1C1C1C;margin-bottom:.3rem">{av_sel}</div>
              <div style="font-size:.78rem;color:#AAA89E;margin-bottom:.6rem">
                {av_row.get("brand_name","")} &nbsp;·&nbsp; {av_row.get("country_name","")} &nbsp;·&nbsp;
                {av_row.get("campaign_display_name","")}
              </div>
              {av_link}
              {caption_html}
              <div style="margin-top:1rem">
                <div style="font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.4rem">Binary features present</div>
                <div style="font-size:.83rem;color:#3A3830">
                  {", ".join(av_feats_bin) if av_feats_bin else "None detected"}
                </div>
              </div>
              <div style="margin-top:.9rem">
                <div style="font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.4rem">All features</div>
                {feat_rows_html}
              </div>
              {obj_html_av}
            </div>""", unsafe_allow_html=True)

        with av_col2:
            # score cards for this asset
            st.markdown('<div style="font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.3rem">Scores for this asset</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="def-note">“vs scope” = this asset minus the average of the {len(sub_df):,} assets matching your filters. '
                        'Tiredness: lower is better.</div>', unsafe_allow_html=True)
            for ml, mc_key, mc_col in [
                ("Attention",        "Attention_T2B",             BLUE),
                ("Persuasion",       "Persuasion_T2B",            RED),
                ("Likeability",      "Likeability_Love_Like_T2B", GREEN),
                ("Experience Recall","Experience_Recall_T2B",     "#B45309"),
                ("Brand Linkage",    "Brand_Linkage_T2B",         "#0E7490"),
                ("Uniqueness",       "Uniqueness_T2B",             "#6D28D9"),
                ("Shareability",     "Shareability_T2B",           "#065F46"),
                ("Tiredness",        "Tiredness_T2B",              "#9F1239"),
            ]:
                val = av_row.get(mc_key)
                if pd.notna(val):
                    disp = f"{val*100:.1f}pp"
                    scope_mean = sub_df[mc_key].dropna().mean()
                    diff = val - scope_mean
                    diff_disp = f"{diff*100:+.1f}pp vs scope"
                    diff_col = ucol(diff*100, mc_key) if diff != 0 else "#AAA"
                    st.markdown(f"""<div style="display:flex;align-items:center;gap:.8rem;padding:.5rem 0;border-bottom:1px solid #F5F3EF">
                      <div style="width:8px;height:8px;border-radius:50%;background:{mc_col};flex-shrink:0"></div>
                      <div style="flex:1;font-size:.82rem;color:#6A6660">{ml}</div>
                      <div style="font-size:1rem;font-weight:700;color:{mc_col}">{disp}</div>
                      <div style="font-size:.75rem;color:{diff_col};min-width:80px;text-align:right">{diff_disp}</div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "02 · Feature Impact":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Why · Page 2 of 7</div>
      <div class="hero-title">Feature Impact</div>
      <div class="hero-sub">For each creative feature, the statistical uplift on your
        chosen metric — how much higher or lower the score is when the feature is present.
        Use this page to identify <em>which creative choices drove the results</em> you saw on page 01.</div>
    </div>""", unsafe_allow_html=True)
    how_to("""<strong>Analyst workflow:</strong> Select the metric you care most about, then read the table top-to-bottom.
      Focus on <strong>High</strong> reliability findings (*** or ** with 50+ assets each side).
      Medium is directional; Directional should never appear in a brief on its own.
      Use the <strong>Stability check</strong> to see whether a feature's effect holds across OUs, markets and formats,
      and the <strong>per-value impact</strong> to see which specific values (e.g. "Seasonal: Christmas") drive the result.
      Then go to <strong>03 · Combination Explorer</strong> to test combinations.<br><br>
      <strong>How to read the table:</strong> Uplift = average score of assets in your selection that HAVE the feature
      minus those that DON'T (the comparison group). Stars: *** very confident, ** confident, * directional,
      ns not significant. Coverage = share of assets in your selection with the feature.""")

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()

    mc_sel=st.selectbox("Metric to display", METRIC_NAMES, key="fi_mc")
    mc=mc_of(mc_sel)
    tired_note(mc)
    n_scope=len(sub_df)

    col_chart, col_alerts = st.columns([1.6,1])

    with col_chart:
        rows=[]
        for feat in ALL_FEATS:
            if feat not in sub_df.columns: continue
            u,sig,n=compute_uplift(sub_df,feat,mc)
            if u is not None:
                n0 = int((~has_mask(sub_df,feat)).sum())
                rows.append({"feat":feat,"label":FEAT_LABEL.get(feat,feat),"uplift":u,"sig":sig,"n":n,
                             "coverage":n/n_scope,"rel":rel_label(sig,n,n0)})
        if rows:
            fd=pd.DataFrame(rows).sort_values("uplift",ascending=lower_better(mc))
            tbl = """<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.85rem">
            <thead><tr style="border-bottom:2px solid #EAE8E2">
              <th style="text-align:left;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Feature</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Uplift</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Direction</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#7030A0">Sig.</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">n · coverage</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Reliability</th>
            </tr></thead><tbody>"""
            max_abs = fd["uplift"].abs().max() or 1
            for i, (_, r) in enumerate(fd.iterrows()):
                bg = "#FAFAF8" if i%2==0 else "#FFFFFF"
                u2 = r["uplift"]
                uc = ucol(u2, mc)
                arrow = "▲" if u2>=0 else "▼"
                sig2 = r["sig"]
                sig_color = "#7030A0" if sig2 in ("***","**","*") else "#AAA"
                bar_w = int(abs(u2)/max_abs*80)
                bar_html_str = (f'<div style="display:flex;align-items:center;gap:.3rem;justify-content:{"flex-start" if u2>=0 else "flex-end"}">'
                                f'<div style="width:{bar_w}px;height:6px;background:{uc};border-radius:2px;opacity:.7"></div>'
                                f'</div>')
                desc = FEAT_DESC.get(r["feat"], "")
                tbl += (f'<tr style="background:{bg};border-bottom:1px solid #F0EEE8">'
                        f'<td style="padding:.42rem .6rem"><div style="font-weight:600;color:#1C1C1C">{r["label"]}</div>'
                        f'<div style="font-size:.7rem;color:#AAA;line-height:1.35">{desc}</div></td>'
                        f'<td style="text-align:center;font-weight:700;color:{uc};font-size:.9rem">{arrow} {abs(u2):.1f}pp</td>'
                        f'<td style="padding:.42rem .2rem">{bar_html_str}</td>'
                        f'<td style="text-align:center;font-weight:600;color:{sig_color}">{sig2}</td>'
                        f'<td style="text-align:center;color:#888">{r["n"]:,} · {r["coverage"]:.0%}</td>'
                        f'<td style="text-align:center">{rel_badge(r["rel"])}</td>'
                        f'</tr>')
            tbl += "</tbody></table></div>"
            tbl += (f'<div style="font-size:.73rem;color:#AAA;margin-top:.4rem">Sorted by {mc_sel} uplift'
                    f'{" (most helpful = lowest Tiredness first)" if lower_better(mc) else ""}. '
                    f'pp = percentage points. *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, ns = not significant.</div>')
            st.markdown(tbl, unsafe_allow_html=True)
            dl(fd.assign(description=fd["feat"].map(FEAT_DESC), metric=mc_sel)
                 .rename(columns={"label":"Feature","uplift":"Uplift (pp)","sig":"Sig.","n":"n with feature",
                                  "coverage":"Coverage","rel":"Reliability"}).drop(columns="feat"),
               "feature_impact", "fi_tbl")
        else:
            st.info("Not enough data to compute feature impact for this scope.")

    with col_alerts:
        st.markdown('<div class="sec-label">Findings summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="def-note"><strong>Uplift is measured against assets without the feature</strong> — '
                    f'the average {mc_sel} of assets in your selection that do <em>not</em> have it (the control group). '
                    f'It is not compared with the overall average.</div>', unsafe_allow_html=True)

        sigrows = [r for r in (rows or []) if r["sig"] in ("***","**","*")]
        pos_feats = sorted([r for r in sigrows if is_good(r["uplift"], mc)], key=lambda r: -abs(r["uplift"]))
        neg_feats = sorted([r for r in sigrows if not is_good(r["uplift"], mc)], key=lambda r: -abs(r["uplift"]))

        def summary_rows(items, color):
            if not items:
                return '<div style="font-size:.79rem;color:#AAA;padding:.3rem 0">No significant findings</div>'
            html = ""
            for r in items[:5]:
                html += (f'<div style="padding:.28rem 0;border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                         f'<div style="display:flex;justify-content:space-between">'
                         f'<span style="color:#1C1C1C;font-weight:600">{r["label"]}</span>'
                         f'<span><span style="font-weight:600;color:{color}">{r["uplift"]:+.1f}pp</span> {rel_badge(r["rel"])}</span></div>'
                         f'<div style="font-size:.7rem;color:#999">{FEAT_DESC.get(r["feat"],"")}</div></div>')
            return html

        st.markdown(f"""
        <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
          color:#2A8050;margin:.5rem 0 .2rem 0">Features improving {mc_sel}</div>
        {summary_rows(pos_feats, GREEN)}
        <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
          color:#C00020;margin:.8rem 0 .2rem 0">Features worsening {mc_sel}</div>
        {summary_rows(neg_feats, RED)}
        <div style="font-size:.69rem;color:#AAA;margin-top:.7rem">Significant features only (*** ** *).</div>""",
        unsafe_allow_html=True)

        # Market level of the KPI (not feature-specific)
        if "country_name" in sub_df.columns and sub_df["country_name"].nunique() > 1:
            st.markdown('<div class="sec-label" style="margin-top:1.4rem">Markets vs your selection average</div>', unsafe_allow_html=True)
            scope_mean_mc = sub_df[mc].dropna().mean() * 100
            st.markdown(f'<div class="def-note">This is about the <strong>{mc_sel} score itself</strong>, not about any feature: '
                        f'the average {mc_sel} of all assets in your selection in each market, compared with your whole '
                        f'selection ({scope_mean_mc:.1f}pp). Use it to see where performance is stronger before reading the features.</div>',
                        unsafe_allow_html=True)
            mks = sub_df.groupby("country_name")[mc].agg(["mean","count"]).dropna()
            mks = pd.DataFrame({"Market": mks.index, "Score (pp)": mks["mean"].values*100, "n assets": mks["count"].values})
            mks["vs selection (pp)"] = mks["Score (pp)"] - scope_mean_mc
            better = mks[mks["vs selection (pp)"].apply(lambda v: is_good(v, mc) and abs(v) > 1)]
            worse  = mks[mks["vs selection (pp)"].apply(lambda v: (not is_good(v, mc)) and abs(v) > 1)]
            for title, d, c in [("Better than your selection", better, GREEN), ("Worse than your selection", worse, RED)]:
                st.markdown(f'<div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:{c};margin:.6rem 0 .2rem 0">{title}</div>', unsafe_allow_html=True)
                if d.empty:
                    st.markdown('<div style="font-size:.79rem;color:#AAA;padding:.3rem 0">No markets beyond ±1pp</div>', unsafe_allow_html=True)
                for _, r in d.sort_values("vs selection (pp)", key=abs, ascending=False).head(5).iterrows():
                    st.markdown(f'<div style="display:flex;justify-content:space-between;padding:.28rem 0;border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                                f'<span style="color:#1C1C1C">{r["Market"]} <span style="color:#CCC;font-size:.72rem">n={int(r["n assets"])}</span></span>'
                                f'<span style="font-weight:600;color:{c}">{r["vs selection (pp)"]:+.1f}pp</span></div>', unsafe_allow_html=True)
            dl(mks.round(2), "markets_vs_selection", "fi_mkt")

    # ── Stability check ───────────────────────────────────────────────────────
    if rows:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Stability check — does the effect hold everywhere?</div>', unsafe_allow_html=True)
        _st_opts = [r["feat"] for r in sorted(rows, key=lambda r: -abs(r["uplift"]))]
        _st_f = st.selectbox("Feature", _st_opts, format_func=lambda f: FEAT_LABEL.get(f,f), key="fi_stab")
        render_stability(sub_df, has_mask(sub_df, _st_f), mc, FEAT_LABEL.get(_st_f,_st_f), "fi_stab")

    # ── Per-value breakdown for categorical features ──────────────────────────
    with st.expander("Categorical features — per-value impact (e.g. which Seasonal or Occasion value)", expanded=False):
        st.markdown("""<div style="font-size:.82rem;color:#555;margin-bottom:.7rem">
        Each individual value is tested against assets that have <em>no</em> value for that feature
        (for features every asset has, such as Color Contrast, against all other assets).
        This shows which specific values drive performance, not just whether the feature is present.</div>""",
        unsafe_allow_html=True)
        _cat_opts = [f for f in ALL_FEATS if f not in BIN_FEATS and f in sub_df.columns]
        if _cat_opts:
            _pv_f = st.selectbox("Feature", _cat_opts, format_func=lambda f: FEAT_LABEL.get(f,f),
                                 index=_cat_opts.index("occasions") if "occasions" in _cat_opts else 0, key="fi_pv")
            st.markdown(f'<div style="font-size:.75rem;color:#888;margin-bottom:.35rem">{FEAT_DESC.get(_pv_f,"")}</div>', unsafe_allow_html=True)
            pv_rows = value_uplifts(sub_df, _pv_f, mc, min_n=5)[:15]
            if pv_rows:
                tbl_pv = '<table style="width:100%;border-collapse:collapse;font-size:.78rem">'
                tbl_pv += '<thead><tr style="border-bottom:1px solid #EAE8E2"><th style="text-align:left;padding:.3rem .4rem;color:#888;font-size:.68rem">Value</th><th style="text-align:center;padding:.3rem .4rem;color:#888;font-size:.68rem">Uplift</th><th style="text-align:center;color:#7030A0;font-size:.68rem">Sig</th><th style="text-align:center;color:#888;font-size:.68rem">n</th><th style="text-align:center;color:#888;font-size:.68rem">Reliability</th></tr></thead><tbody>'
                for idx_pv, rv in enumerate(pv_rows):
                    uc2 = ucol(rv["uplift"], mc)
                    bg2 = "#FAFAF8" if idx_pv%2==0 else "#FFF"
                    sc2 = "#7030A0" if rv["sig"] in ("***","**","*") else "#CCC"
                    tbl_pv += (f'<tr style="background:{bg2};border-bottom:1px solid #F5F3EF">'
                               f'<td style="padding:.28rem .4rem;max-width:300px">{rv["value"][:50]}</td>'
                               f'<td style="text-align:center;font-weight:600;color:{uc2}">{rv["uplift"]:+.1f}pp</td>'
                               f'<td style="text-align:center;color:{sc2}">{rv["sig"]}</td>'
                               f'<td style="text-align:center;color:#AAA">{rv["n"]}</td>'
                               f'<td style="text-align:center">{rel_badge(rel_label(rv["sig"],rv["n"],rv["n_ref"]))}</td></tr>')
                tbl_pv += "</tbody></table>"
                st.markdown(tbl_pv, unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:.72rem;color:#AAA;margin-top:.3rem">Compared with {pv_rows[0]["ref"]}.</div>', unsafe_allow_html=True)
                dl(pd.DataFrame(pv_rows).assign(feature=FEAT_LABEL.get(_pv_f,_pv_f), metric=mc_sel),
                   f"value_impact_{_pv_f}", "fi_pv")
            else:
                st.markdown('<p style="color:#AAA;font-size:.8rem">Not enough data.</p>', unsafe_allow_html=True)

    # Insight catalog entries for this metric
    if has_ins and catalog is not None:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Insight catalog — top findings for this metric</div>',
                    unsafe_allow_html=True)
        cat_view=catalog[catalog["metric_display"]==mc_sel].copy()
        _cv_scope = "Global (all assets)"
        if scope_filters:
            sl=SMAP.get(scope_filters[0][0],scope_filters[0][0]); sv=scope_filters[0][1]
            _cv = cat_view[(cat_view["filter"]==sl)&(cat_view["filter_value"]==sv)]
            if len(_cv): cat_view, _cv_scope = _cv, f"{sl} = {sv}"
            else: cat_view = cat_view[cat_view["filter"]=="Global"]
        else:
            cat_view=cat_view[cat_view["filter"]=="Global"]
        st.markdown(f"""<div class="explain-box">
        Validated findings from the full insight pipeline for <strong>{_cv_scope}</strong>
        (the pipeline covers single OUs, brands and markets; year and campaign filters do not apply here).
        <strong>Baseline</strong> is the average score of assets that do <em>not</em> have the feature.</div>""",
        unsafe_allow_html=True)
        cat_view=cat_view[cat_view["confidence"].isin(["high","medium"])].sort_values(
            "evidence_uplift_pp",ascending=lower_better(mc))

        for _,r in cat_view.head(12).iterrows():
            v=r["evidence_uplift_pp"]; bc=ucol(v, mc)
            st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
              <div style="display:flex;justify-content:space-between">
                <div class="ic-feat">{r["feature_display"]}</div>
                <div>{badge(f'{"▲" if v>=0 else "▼"} {abs(v):.1f}pp', "b-pos" if is_good(v, mc) else "b-neg")}</div>
              </div>
              <div class="ic-text">{r["text"]}</div>
              <div style="font-size:.72rem;color:#999;margin-top:3px">{FEAT_DESC.get(r["feature"],"")}</div>
              <div class="ic-meta">
                {conf_badge(r["confidence"])}
                {badge(r["metric_display"],"b-metric")}
                {sig_badge(r["evidence_sig"])}
                <span class="badge b-scope">n={int(r["evidence_n_has"]):,}</span>
                <span class="badge b-ns">base {r["evidence_baseline_pp"]:.1f}pp</span>
              </div>
            </div>""", unsafe_allow_html=True)
        dl(cat_view, "catalog_findings", "fi_cat")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 03 — COMBINATION EXPLORER  (dashboard_app logic)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "03 · Combination Explorer":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Why · Page 3 of 7</div>
      <div class="hero-title">Combination Explorer</div>
      <div class="hero-sub">Features rarely work in isolation. This page finds the sequence of
        features that, combined, maximise a metric — and lets you test any custom combination
        in real time across all 8 metrics simultaneously.</div>
    </div>""", unsafe_allow_html=True)
    how_to("""<strong>What you see per metric tab:</strong> the <strong>best combination</strong> for that metric, chosen
      automatically, plus up to 2 <strong>alternatives</strong> that use different features (in the expander).
      <br><br><strong>How the best combination is chosen:</strong> start from the <strong>baseline</strong> — the average
      score of all assets in your selection — and add one feature value at a time, always the one that moves the score
      most (up; for Tiredness, down). Each step's <strong>gain</strong> is measured against the previous step; the
      <strong>total</strong> is the sum of the step gains, i.e. final score minus baseline. <strong>n</strong> is how many
      assets have every feature so far — smaller n means a less certain estimate.
      <br><br><strong>Explorer:</strong> pre-filled with the best combination. Change any feature to see all 8 metrics
      update; a combination that boosts Attention but raises Tiredness may not be worth briefing.""")

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()
    n_scope = len(sub_df)
    sig_key = hash(CTX["scope_sig"])

    ALL_COMBO_METRICS = [(m, mc_of(m)) for m in METRIC_NAMES]
    tabs = st.tabs([m[0] for m in ALL_COMBO_METRICS])
    for (ml, mc), tab in zip(ALL_COMBO_METRICS, tabs):
        with tab:
            tired_note(mc)
            top3, engine = get_top_combos(sub_df, scope_key, mc)
            top3 = [t for t in top3 if t.get("steps")]
            combo = top3[0]["combo"] if top3 else None

            # ── alerts for this metric (scope + features of the best combination) ──
            al = []
            if top3:
                for stp in top3[0]["steps"]: al += get_feature_alerts(stp["feat"], scope_filters, mc, max_items=2)
            al += get_scope_alerts(scope_filters, mc, max_items=4)
            seen_t = set(); al = [x for x in al if not (x["text"] in seen_t or seen_t.add(x["text"]))]
            if al:
                render_alerts(al, max_items=6, title=f"⚠ Check before briefing — {len(al[:6])} alert{'s' if len(al[:6])>1 else ''} on {ml}")

            if not top3:
                st.markdown('<p style="color:#CCC8C2;padding:1.5rem 0;font-size:.84rem">No combination found for this selection — '
                            'too few assets share any feature value.</p>', unsafe_allow_html=True)
            else:
                best = top3[0]; steps = best["steps"]
                baseline_pp, n_total = best["baseline_pp"], best["n_total"]
                final_mean, total_uplift = steps[-1]["metric_mean"], steps[-1]["cumulative_uplift_pp"]
                m_best = apply_sel(sub_df, default_sel(combo))
                n_in = int(m_best.sum())
                g_in = sub_df.loc[m_best, mc].dropna(); g_out = sub_df.loc[~m_best, mc].dropna()
                sig_c = _sig(g_in, g_out) if len(g_in) >= 3 and len(g_out) >= 3 else "ns"
                rel_c = rel_label(sig_c, len(g_in), len(g_out))
                tot_col = ucol(total_uplift, mc)

                st.markdown(f'<div class="sec-label" style="margin-top:.6rem">Best combination for {ml} '
                            f'<span style="font-weight:400;letter-spacing:0;text-transform:none;color:#AAA">'
                            f'— picked automatically for this metric{" · computed live for your selection" if engine=="live" else ""}</span></div>',
                            unsafe_allow_html=True)
                col_wf, col_right = st.columns([1.5, 1])
                with col_wf:
                    gains = " ".join(f'{s_["step_gain_pp"]:+.1f}' for s_ in steps)
                    rows_html = "".join(
                        f'<tr style="border-bottom:1px solid #F5F3EF"><td style="padding:.38rem .4rem;color:#E8002D;font-weight:600">{s_["step"]}</td>'
                        f'<td style="padding:.38rem .4rem">+ {s_["label"]}</td>'
                        f'<td style="text-align:center;color:{ucol(s_["step_gain_pp"], mc)};font-weight:600">{s_["step_gain_pp"]:+.1f}</td>'
                        f'<td style="text-align:center;font-weight:600">{s_["metric_mean"]:.1f} <span style="color:#AAA;font-weight:400">({s_["cumulative_uplift_pp"]:+.1f})</span></td>'
                        f'<td style="text-align:center;color:#AAA">{s_["n"]:,}</td></tr>' for s_ in steps)
                    st.markdown(f"""
                    <div class="combo-card">
                      <div style="display:flex;align-items:flex-end;gap:.9rem;flex-wrap:wrap">
                        <span class="combo-uplift" style="color:{tot_col}">{total_uplift:+.1f}</span>
                        <span style="font-family:'Merriweather',serif;font-size:1.3rem;color:#D0C8C0;font-weight:300">pp</span>
                        <div style="padding-bottom:.22rem;font-size:.8rem;color:#555;line-height:1.5">
                          <strong>{baseline_pp:.1f}pp → {final_mean:.1f}pp</strong> {ml}<br>
                          Total = sum of step gains ({gains})
                        </div>
                      </div>
                      <div class="def-note" style="margin-top:.7rem"><strong>Baseline {baseline_pp:.1f}pp</strong> = average {ml}
                        of all {n_total:,} assets in your selection, before any feature is chosen.</div>
                      <table style="width:100%;border-collapse:collapse;font-size:.84rem">
                        <thead><tr style="border-bottom:1px solid #EAE8E2;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:#AAA">
                          <th></th><th style="text-align:left;padding:.3rem .4rem">Feature added</th>
                          <th>Gain vs previous step</th><th>Score (total vs baseline)</th><th>n assets</th></tr></thead>
                        <tbody><tr style="border-bottom:1px solid #F5F3EF;color:#AAA"><td style="padding:.38rem .4rem">—</td>
                          <td style="padding:.38rem .4rem;font-style:italic">Baseline — all assets in selection</td><td></td>
                          <td style="text-align:center">{baseline_pp:.1f}</td><td style="text-align:center">{n_total:,}</td></tr>{rows_html}</tbody>
                      </table>
                      <div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.8rem;align-items:center;font-size:.78rem;color:#555">
                        <span class="badge b-scope">n = {n_in:,} assets</span>
                        <span class="badge b-scope">coverage {n_in/max(n_scope,1):.0%} of selection</span>
                        {sig_badge(sig_c)} {rel_badge(rel_c)}
                      </div>
                    </div>""", unsafe_allow_html=True)
                    if n_in < 10:
                        st.markdown(f'<div class="low-n">Only {n_in} assets match — treat as directional.</div>', unsafe_allow_html=True)
                    dl(pd.DataFrame([{"Step":0,"Feature":"Baseline (all assets in selection)","Gain vs previous (pp)":None,
                                      "Score (pp)":baseline_pp,"Total vs baseline (pp)":0,"n":n_total}] +
                                    [{"Step":s_["step"],"Feature":s_["label"],"Gain vs previous (pp)":s_["step_gain_pp"],
                                      "Score (pp)":s_["metric_mean"],"Total vs baseline (pp)":s_["cumulative_uplift_pp"],"n":s_["n"]}
                                     for s_ in steps]).assign(Metric=ml), f"best_combination_{ml}", f"cx_{mc}")

                    if len(top3) > 1:
                        with st.expander(f"{len(top3)-1} alternative combination{'s' if len(top3)>2 else ''} (using different features)"):
                            for alt in top3[1:]:
                                ast_ = alt["steps"]
                                m_alt = apply_sel(sub_df, default_sel(alt["combo"]))
                                a_in = sub_df.loc[m_alt, mc].dropna(); a_out = sub_df.loc[~m_alt, mc].dropna()
                                a_sig = _sig(a_in, a_out) if len(a_in) >= 3 and len(a_out) >= 3 else "ns"
                                st.markdown(f'<div style="padding:.5rem 0;border-bottom:1px solid #F0EEE8;font-size:.84rem">'
                                            f'<strong>{" + ".join(x["label"] for x in ast_)}</strong> → '
                                            f'<span style="color:{ucol(ast_[-1]["cumulative_uplift_pp"], mc)};font-weight:600">'
                                            f'{ast_[-1]["cumulative_uplift_pp"]:+.1f}pp</span> '
                                            f'<span style="color:#AAA">({alt["baseline_pp"]:.1f} → {ast_[-1]["metric_mean"]:.1f}pp)</span> '
                                            f'<span class="badge b-scope">n={int(m_alt.sum()):,}</span> '
                                            f'{rel_badge(rel_label(a_sig, len(a_in), len(a_out)))}</div>', unsafe_allow_html=True)

                with col_right:
                    st.markdown('<div class="sec-label">Top assets matching this combination</div>', unsafe_allow_html=True)
                    top_tmp = sub_df.loc[m_best].sort_values(mc, ascending=lower_better(mc)).head(4)
                    for _, r in top_tmp.iterrows():
                        url=r.get("asset_url",""); nm=r.get("asset_name",f"Asset {r.get('asset_sk_id','')}")
                        lnk=f'<a href="{url}" target="_blank" style="color:{RED};font-size:.71rem;font-weight:600;text-decoration:none">▶ View</a>' if url else ""
                        sc=f"{r[mc]*100:.0f}pp" if pd.notna(r.get(mc)) else "—"
                        st.markdown(f"""<div class="asset-card">
                          <div style="display:flex;justify-content:space-between">
                            <div class="asset-name" style="font-size:.76rem">{str(r.get("campaign_display_name",""))[:40]} · {r.get("country_name","")}</div>{lnk}
                          </div>
                          <div class="asset-meta">{ml} {sc} &nbsp;·&nbsp; {str(nm)[:45]}</div>
                        </div>""", unsafe_allow_html=True)
                    with st.expander("Stability check — does it hold across OUs, markets, formats?"):
                        render_stability(sub_df, m_best, mc, "this combination", f"cx_st_{mc}")

            # ── Feature Explorer ─────────────────────────────────────────────
            st.markdown("""<div class="explorer-card">
              <div class="explorer-title">Explore any combination</div>
              <div class="explorer-sub">
                Pre-filled with the best combination above. Change any feature to see all
                metrics update in real time. Each score is the average of the assets that match
                your selection; the delta compares it with all assets in your filters.
              </div>""",unsafe_allow_html=True)

            ss=f"sel_{mc}_{sig_key}"
            if ss not in st.session_state:
                st.session_state[ss]=default_sel(combo) if combo else {f:"__any__" for f in ALL_FEATS}
            cur=st.session_state[ss]; new=dict(cur); changed=False

            st.markdown('<div class="feat-group">Asset features — select a value to filter</div>',unsafe_allow_html=True)
            rows_f=[ALL_FEATS[i:i+5] for i in range(0,len(ALL_FEATS),5)]
            for row_feats in rows_f:
                cols=st.columns(5)
                for ci,feat in enumerate(row_feats):
                    with cols[ci]:
                        if feat not in sub_df.columns: new[feat]="__any__"; continue
                        raws,disps=get_opts(feat,scope_key,mc,sub_df)
                        cr=cur.get(feat,"__any__")
                        if cr not in raws: raws, disps = raws+[cr], disps+[str(cr)]
                        ci2=raws.index(cr)
                        ch=st.selectbox(FEAT_LABEL.get(feat,feat),range(len(raws)),
                                         format_func=lambda i,d=disps:d[i],index=ci2,
                                         key=f"d_{mc}_{feat}_{sig_key}")
                        chv=raws[ch]; new[feat]=chv
                        if chv!=cr: changed=True
                for ci in range(len(row_feats),5): cols[ci].empty()

            if changed: st.session_state[ss]=new; st.rerun()

            r1,r2,_=st.columns([1,1,7])
            with r1:
                if st.button("↺ Reset",key=f"rst_{mc}_{sig_key}"):
                    st.session_state[ss]=default_sel(combo) if combo else {f:"__any__" for f in ALL_FEATS}
                    st.rerun()
            with r2:
                if st.button("✕ Clear",key=f"clr_{mc}_{sig_key}"):
                    st.session_state[ss]={f:"__any__" for f in ALL_FEATS}; st.rerun()

            # Scoreboard
            scores=score_sel(sub_df,cur)
            bases={mc2:round(sub_df[mc2].dropna().mean()*100,2) for mc2 in ALL_METRICS if mc2 in sub_df.columns}
            sb='<div class="scoreboard">'; sb_rows=[]
            for mc2 in ALL_METRICS:
                if mc2 not in sub_df.columns: continue
                ml2 = M_LABEL.get(mc2, METRICS.get(mc2, mc2))
                val,nv=scores.get(mc2,(None,0)); base=bases.get(mc2,0)
                hi=" hi" if mc2==mc else ""; gc=M_COLOR.get(mc2,DARK)
                if val is not None:
                    d=val-base; dc="d-flat" if abs(d)<=.1 else ("d-up" if is_good(d, mc2) else "d-dn")
                    sb+=(f'<div class="score-cell{hi}" style="border-top:2px solid {gc}">'
                         f'<div class="score-metric">{ml2}{" ↓" if lower_better(mc2) else ""}</div>'
                         f'<div class="score-val">{val:.1f}</div>'
                         f'<div class="score-delta {dc}">{"+" if d>0 else ""}{d:.1f}pp vs scope</div>'
                         f'<div class="score-n">n={nv:,}</div></div>')
                    sb_rows.append({"Metric":ml2,"Selection score (pp)":val,"Scope average (pp)":base,"Difference (pp)":round(d,2),"n":nv})
                else:
                    sb+=(f'<div class="score-cell{hi}" style="border-top:2px solid {gc}">'
                         f'<div class="score-metric">{ml2}</div>'
                         f'<div class="score-val" style="font-size:.9rem;color:#D8D4CE">—</div>'
                         f'<div class="score-n">n=0</div></div>')
            st.markdown(sb+"</div>",unsafe_allow_html=True)
            st.markdown('<div style="font-size:.72rem;color:#AAA;margin-top:.35rem">↓ = lower is better (Tiredness). '
                        'Green = better than all assets in your filters, red = worse.</div>', unsafe_allow_html=True)
            dl(pd.DataFrame(sb_rows).assign(Selection=", ".join(f"{FEAT_LABEL.get(f,f)}={v}" for f,v in cur.items() if v!="__any__") or "Any"),
               f"explorer_scores_{ml}", f"cx_sb_{mc}")

            _,mn=scores.get(mc,(None,0))
            if mn is not None and 0<mn<10:
                st.markdown(f'<div class="low-n">Only {mn} assets match — directional only.</div>',unsafe_allow_html=True)

            # Insights for active selection
            active_sel=[f for f,v in cur.items() if v not in ("__any__",None,"")]
            if active_sel:
                items=[]
                for f in active_sel[:4]: items+=get_feature_alerts(f,scope_filters,mc,max_items=2)
                if items:
                    st.markdown('<div style="margin-top:.9rem;border-top:1px solid #EAE8E2;padding-top:.9rem">'
                                f'<div class="sec-label">Alerts for your selection — {ml}</div>',unsafe_allow_html=True)
                    render_alerts(items[:5]); st.markdown("</div>",unsafe_allow_html=True)

            st.markdown("</div>",unsafe_allow_html=True)  # close explorer-card


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 04 — FEATURE COMBINATIONS & OU IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "04 · Feature Combinations & OU Impact":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Why · Page 4 of 7</div>
      <div class="hero-title">Feature Combinations &amp; OU Impact</div>
      <div class="hero-sub">Which features amplify each other, and does the same feature work in every
        Operating Unit? A feature that is positive on average can reverse in specific OUs.</div>
    </div>""", unsafe_allow_html=True)
    how_to("""<strong>Workflow:</strong> pick the feature you want to brief and your priority metric. Read the
      <strong>So what</strong> box first, then the alerts. Use <em>Best partners</em> to see what to pair it with and
      <em>By OU</em> to see where it reverses — never brief a feature without checking the OU view.
      <br><br><strong>Combined uplift</strong> = average score of assets with <em>both</em> features minus the average of
      all assets in your selection. <strong>Partner alone</strong> = same for the partner on its own.
      <strong>Synergy</strong> = combined uplift minus the better of the two features alone (positive = they amplify each
      other). <strong>Significance</strong> compares assets with both features against all other assets in your selection.
      <strong>OU uplift</strong> = assets with the feature vs without it, inside each OU (your Brand/Year/Market filters still apply).""")

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()

    feat_opts=[(f,FEAT_LABEL.get(f,f)) for f in BIN_FEATS if f in sub_df.columns]
    feat_labels=[fl for _,fl in feat_opts]
    feat_cols=[fc for fc,_ in feat_opts]
    s1, s2 = st.columns(2)
    with s1:
        fi=st.selectbox("Select a feature to analyse",range(len(feat_labels)),
                         format_func=lambda i:feat_labels[i],key="combo_feat")
    with s2:
        mc_c=st.selectbox("Metric", METRIC_NAMES, key="combo_mc")
    sel_f=feat_cols[fi]; sel_fl=feat_labels[fi]
    mc_c_col=mc_of(mc_c)
    tired_note(mc_c_col)

    # Solo uplift
    u_s,sig_s,n_s=compute_uplift(sub_df,sel_f,mc_c_col)
    combos=feature_combinations(sub_df,sel_f,mc_c_col,top_n=7)

    # OU rows (scope filters other than OU still apply; year & campaign via sub_df's other filters)
    ou_base = df_full[df_full["_year"].isin(CTX["years"])] if CTX["years"] else df_full
    for t,v in scope_filters:
        if t=="brand": ou_base = ou_base[ou_base["brand_name"]==v]
    if any(t=="market" for t,_ in scope_filters):
        ou_base = ou_base[ou_base["country_name"].isin([v for t,v in scope_filters if t=="market"])]
    ou_rows=[]
    for ou in sorted(ou_base["operating_unit_code"].dropna().unique()):
        ou_sub=ou_base[ou_base["operating_unit_code"]==ou]
        u2,sig2,n2=compute_uplift(ou_sub,sel_f,mc_c_col)
        if u2 is not None:
            ou_rows.append({"OU":ou,"uplift":u2,"sig":sig2,"n":n2,
                            "rel":rel_label(sig2,n2,int((~has_mask(ou_sub,sel_f)).sum()))})

    # ── So what ───────────────────────────────────────────────────────────────
    bullets=[]
    if u_s is not None:
        n0_s=int((~has_mask(sub_df,sel_f)).sum())
        bullets.append(f"<strong>{sel_fl}</strong> {'improves' if is_good(u_s,mc_c_col) else 'worsens'} {mc_c} by "
                       f"<strong>{u_s:+.1f}pp</strong> in your selection ({sig_s}, n={n_s:,}, "
                       f"{n_s/len(sub_df):.0%} of assets) — {rel_badge(rel_label(sig_s,n_s,n0_s))}")
    if combos:
        c0=combos[0]
        bullets.append(f"Best partner: <strong>{c0['label']}</strong> — together {c0['combined']:+.1f}pp vs your selection "
                       f"average ({c0['sig']}, n={c0['n']:,}); synergy {c0['synergy']:+.1f}pp "
                       f"({'they amplify each other' if is_good(c0['synergy'],mc_c_col) else 'little extra from pairing'}) — "
                       f"{rel_badge(rel_label(c0['sig'],c0['n'],c0['n_rest']))}")
    if ou_rows:
        od=pd.DataFrame(ou_rows)
        od_s=od[od["sig"].isin(["*","**","***"])]
        pick=od_s if len(od_s) else od
        bst=pick.sort_values("uplift",ascending=lower_better(mc_c_col)).iloc[0]
        wst=pick.sort_values("uplift",ascending=not lower_better(mc_c_col)).iloc[0]
        rev=od[(od["sig"].isin(["*","**","***"])) & (od["uplift"].apply(lambda v: not is_good(v,mc_c_col)))]
        txt=(f"Works best in <strong>{bst['OU']}</strong> ({bst['uplift']:+.1f}pp, {bst['sig']}); "
             f"weakest in <strong>{wst['OU']}</strong> ({wst['uplift']:+.1f}pp, {wst['sig']}).")
        txt+=(f" <strong>Significantly reverses in {', '.join(rev['OU'])}</strong> — adapt the brief there."
              if len(rev) else " No OU shows a significant reversal.")
        bullets.append(txt)
    if bullets:
        st.markdown('<div class="takeaway"><div class="takeaway-title">So what</div><ul>'
                    + "".join(f"<li>{b_}</li>" for b_ in bullets) + "</ul></div>", unsafe_allow_html=True)

    fit=get_feature_alerts(sel_f,scope_filters,mc_c_col,max_items=5)
    if fit:
        render_alerts(fit, max_items=5, title=f"⚠ Alerts for {sel_fl} on {mc_c} — read before briefing")

    tab_a, tab_b = st.tabs([f"Best partners for {sel_fl}", f"{sel_fl} by OU"])

    with tab_a:
        if combos:
            tbl="""<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.83rem">
            <thead><tr style="border-bottom:2px solid #EAE8E2">
              <th style="text-align:left;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Partner feature</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Combined uplift</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#7030A0">Sig.</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Partner alone</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Synergy</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">n · coverage</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Reliability</th>
            </tr></thead><tbody>"""
            for ci,c in enumerate(combos):
                bg="#FAFAF8" if ci%2==0 else "#FFF"
                cu=c["combined"]; sy=c["synergy"]
                tbl+=(f'<tr style="background:{bg};border-bottom:1px solid #F0EEE8">'
                      f'<td style="padding:.42rem .6rem;font-weight:600">{c["label"]}</td>'
                      f'<td style="text-align:center;font-weight:600;color:{ucol(cu,mc_c_col)}">{cu:+.1f}pp</td>'
                      f'<td style="text-align:center;color:{"#7030A0" if c["sig"]!="ns" else "#AAA"}">{c["sig"]}</td>'
                      f'<td style="text-align:center;color:#555">{c["solo_p"]:+.1f}pp</td>'
                      f'<td style="text-align:center;font-weight:600;color:{ucol(sy,mc_c_col)}">{sy:+.1f}pp</td>'
                      f'<td style="text-align:center;color:#AAA">{c["n"]} · {c["coverage"]:.0%}</td>'
                      f'<td style="text-align:center">{rel_badge(rel_label(c["sig"],c["n"],c["n_rest"]))}</td></tr>')
            tbl+="</tbody></table></div>"
            st.markdown(tbl,unsafe_allow_html=True)
            dl(pd.DataFrame(combos).assign(feature=sel_fl, metric=mc_c,
                                           reliability=[rel_label(c["sig"],c["n"],c["n_rest"]) for c in combos])
                 .drop(columns=["partner"]).rename(columns={"label":"partner"}), f"partners_{sel_f}", "fc_tbl")
        else:
            st.info("Not enough data to compute combinations in this scope.")

    with tab_b:
        if ou_rows:
            ou_df=pd.DataFrame(ou_rows).sort_values("uplift")
            g_u,_,_=compute_uplift(ou_base,sel_f,mc_c_col)
            fig,ax=plt.subplots(figsize=(7,max(3.5,len(ou_df)*.5)))
            fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
            ax.barh(ou_df["OU"],ou_df["uplift"],
                     color=[GREEN if is_good(v,mc_c_col) else RED for v in ou_df["uplift"]],
                     alpha=.85,height=.62)
            ax.axvline(0,color=DARK,lw=1.2,ls="--")
            if g_u is not None:
                ax.axvline(g_u,color=BLUE,lw=1.5,ls=":",alpha=.75,label=f"All OUs: {g_u:+.1f}pp")
                ax.legend(fontsize=8)
            for i,(_,r) in enumerate(ou_df.iterrows()):
                v=r["uplift"]
                ax.text(v+(.12 if v>=0 else -.12),i,f"{v:+.1f} {r['sig']}  n={r['n']}",
                        va="center",ha="left" if v>=0 else "right",fontsize=8,color=DARK)
            ax.set_xlabel(f"{sel_fl} uplift on {mc_c} (pp)",fontsize=8.5)
            ax.set_title(f"{sel_fl} — {mc_c} by OU",fontsize=9,fontweight="bold",color=DARK)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#DDD")
            ax.grid(axis="x",alpha=.3); plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.markdown('<div style="font-size:.72rem;color:#AAA">Stars on each bar = significance within that OU '
                        '(ns = not significant). Red = the feature worsens the metric in that OU.</div>', unsafe_allow_html=True)

            # best partner per OU (selected metric)
            st.markdown(f'<div class="sec-label" style="margin-top:1rem">Best partner per OU for {sel_fl} — {mc_c}</div>',
                        unsafe_allow_html=True)
            ou_combo_rows=[]
            for ou in sorted(ou_base["operating_unit_code"].dropna().unique()):
                ou_sub2=ou_base[ou_base["operating_unit_code"]==ou]
                cc=feature_combinations(ou_sub2,sel_f,mc_c_col,top_n=1)
                if cc:
                    ou_combo_rows.append({"OU":ou,"Partner":cc[0]["label"],
                                          "Combined uplift (pp)":cc[0]["combined"],
                                          "Sig.":cc[0]["sig"],
                                          "Synergy (pp)":cc[0]["synergy"],
                                          "n":cc[0]["n"],
                                          "Reliability":rel_label(cc[0]["sig"],cc[0]["n"],cc[0]["n_rest"])})
            if ou_combo_rows:
                _ocr=pd.DataFrame(ou_combo_rows)
                st.dataframe(_ocr, hide_index=True, use_container_width=True)
                dl(_ocr.assign(feature=sel_fl, metric=mc_c), f"best_partner_by_ou_{sel_f}", "fc_ou2")
            dl(ou_df.rename(columns={"uplift":"Uplift (pp)","sig":"Sig.","rel":"Reliability"})
                    .assign(feature=sel_fl, metric=mc_c), f"ou_impact_{sel_f}", "fc_ou")
        else:
            st.info("Not enough data for OU breakdown.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 05 — INSIGHT CATALOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "05 · Insight Catalog":
    if not has_ins or catalog is None:
        st.warning("insight_catalog.csv not found. Run the insight pipeline first."); st.stop()

    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">So What · Page 5 of 7</div>
      <div class="hero-title">Insight Catalog</div>
      <div class="hero-sub">The evidence base for creative briefs: every statistically tested finding.
        Start with the Key takeaways — the table below is the full detail.</div>
    </div>""", unsafe_allow_html=True)
    how_to("""<strong>How to use this page in 3 steps:</strong><br>
      1. Read the <strong>Key takeaways</strong> — the few findings worth acting on for your scope.<br>
      2. Keep <strong>Reliability: High</strong> for briefs; Medium is directional; Directional is a hypothesis.<br>
      3. Before using a finding for a specific market, open <strong>Market impact</strong> to check it doesn't reverse there.
      <br><br><strong>Uplift (pp)</strong> — average score of assets with the feature minus assets without it.
      <strong>Baseline</strong> — the average score of assets without the feature.
      <strong>Significance</strong> — *** less than 1-in-1,000 chance the result is noise, ** 1-in-100, * 1-in-20, ns not significant.
      <strong>Quality flags</strong> — small_sample (under 30 assets on one side), multiple_testing_risk (many features
      tested at once), low_value_presence (very few assets have the feature).
      <br><br>The catalog is precomputed for the whole dataset and for single OUs, brands and markets — the Year and
      Campaign filters do not change it.""")

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()

    # Filters — Scope limited to Brand and OU only
    cf1,cf2,cf3,cf4,cf5 = st.columns(5)
    with cf1:
        default_scope="All"
        if scope_filters:
            smap2={"ou":"OU","brand":"Brand","market":"Market"}
            sl=smap2.get(scope_filters[0][0],""); sv=scope_filters[0][1]
            if sl: default_scope=f"{sl}: {sv}"
        scope_opts=["All","Global"]+sorted(
            [f"{r['filter']}: {r['filter_value']}" for _,r in
             catalog[["filter","filter_value"]].drop_duplicates().iterrows()
             if r["filter"] in ("OU","Brand","Market")])
        cat_scope=st.selectbox("Scope",scope_opts,
                                index=scope_opts.index(default_scope) if default_scope in scope_opts else 0)
    with cf2:
        cat_metric=st.multiselect("Metric",sorted(catalog["metric_display"].unique()),
                                   default=sorted(catalog["metric_display"].unique()))
    with cf3:
        cat_rel=st.multiselect("Reliability",["High","Medium","Directional"],default=["High","Medium"])
    with cf4:
        cat_dir=st.selectbox("Effect",["Both","Positive","Negative"])
    with cf5:
        cat_sort=st.selectbox("Sort",["Largest effect","Highest uplift","Lowest uplift","Reliability","Feature"])

    fc=catalog.copy()
    fc["reliability"]=fc["confidence"].map(rel_from_conf)
    fc["coverage"]=fc["evidence_n_has"]/(fc["evidence_n_has"]+fc["evidence_n_abs"]).replace(0,np.nan)
    fc["_good"]=[is_good(v, mc_of(m)) for v,m in zip(fc["evidence_uplift_pp"], fc["metric_display"])]
    if default_scope!="All" and default_scope not in scope_opts:
        st.markdown(f'<div class="low-n">No precomputed findings for {default_scope} (too few assets in the pipeline) — '
                    f'showing all scopes instead.</div>', unsafe_allow_html=True)
    if cat_scope=="Global": fc=fc[fc["filter"]=="Global"]
    elif cat_scope!="All":
        parts=cat_scope.split(": ",1)
        if len(parts)==2: fc=fc[(fc["filter"]==parts[0])&(fc["filter_value"]==parts[1])]
    if cat_metric: fc=fc[fc["metric_display"].isin(cat_metric)]
    if cat_rel:    fc=fc[fc["reliability"].isin(cat_rel)]
    if cat_dir=="Positive": fc=fc[fc["_good"]]
    elif cat_dir=="Negative": fc=fc[~fc["_good"]]
    if cat_sort=="Largest effect": fc=fc.sort_values("evidence_uplift_pp",key=abs,ascending=False)
    elif cat_sort=="Highest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=False)
    elif cat_sort=="Lowest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=True)
    elif cat_sort=="Reliability":
        fc["_co"]=fc["reliability"].map({"High":0,"Medium":1,"Directional":2})
        fc=fc.sort_values(["_co","evidence_uplift_pp"],ascending=[True,False])
    elif cat_sort=="Feature": fc=fc.sort_values("feature_display")

    # ── Key takeaways ─────────────────────────────────────────────────────────
    _ord={"High":0,"Medium":1,"Directional":2}
    _k=fc.assign(_r=fc["reliability"].map(_ord), _a=fc["evidence_uplift_pp"].abs())
    _k=_k[~_k["insight_key"].astype(str).str.contains("Combination", case=False)]
    use=_k[_k["_good"]].sort_values(["_r","_a"],ascending=[True,False]).drop_duplicates(["feature","metric_display"]).head(3)
    avoid=_k[~_k["_good"]].sort_values(["_r","_a"],ascending=[True,False]).drop_duplicates(["feature","metric_display"]).head(2)
    def _tk(r):
        where = "" if r["filter"]=="Global" else f" in {r['filter']} {r['filter_value']}"
        return (f"<strong>{r['feature_display']}</strong> → {r['metric_display']} "
                f"<strong>{r['evidence_uplift_pp']:+.1f}pp</strong>{where} "
                f"(n={int(r['evidence_n_has']):,}) {rel_badge(r['reliability'])}")
    tk=[f"<strong>{len(fc):,} findings</strong> match these filters — "
        f"{int((fc['reliability']=='High').sum())} High, {int((fc['reliability']=='Medium').sum())} Medium."]
    if len(use):   tk.append("<strong>Use:</strong><br>" + "<br>".join(_tk(r) for _,r in use.iterrows()))
    if len(avoid): tk.append("<strong>Avoid:</strong><br>" + "<br>".join(_tk(r) for _,r in avoid.iterrows()))
    if rulebook is not None and len(use):
        _rev=rulebook[(rulebook["rule_type"]=="Heterogeneity") & (rulebook["feature"].isin(use["feature"]))]
        if len(_rev):
            _nf=_rev['feature'].nunique()
            tk.append(f"⚠ {_nf} of the features above {'reverses' if _nf==1 else 'reverse'} in "
                      f"{_rev['scope_value'].nunique()} OU/brand/market scope(s) — check <strong>06 · Rulebook</strong> before briefing locally.")
    st.markdown('<div class="takeaway"><div class="takeaway-title">Key takeaways</div><ul>'
                + "".join(f"<li>{t}</li>" for t in tk) + "</ul></div>", unsafe_allow_html=True)

    # ── Asset Feature Panel ───────────────────────────────────────────────────
    with st.expander("Feature prevalence in your selection — how common is each feature?", expanded=False):
        af_col1, af_col2 = st.columns(2)
        BIN_LABEL_05 = {
            "animals_and_pets_presence":"Animals & Pets","animatics_cartoons_presence":"Animatics / Cartoons",
            "food_presence":"Food","human_presence":"Human","outdoors":"Outdoors",
            "indoors":"Indoors","product_presence":"Product",
        }
        CAT_LABEL_05 = {
            "color_contrast_cat":"Color Contrast","text_color_contrast_cat":"Text Contrast",
            "color_spectrum":"Color Spectrum","tone":"Tone","design_style":"Design Style",
            "occasions":"Occasions","passion_point":"Passion Point","moments":"Moments",
            "music_style":"Music Style","seasonal":"Seasonal",
        }
        with af_col1:
            st.markdown('<div style="font-size:.72rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.6rem">Binary features</div>', unsafe_allow_html=True)
            for col_f, lbl_f in BIN_LABEL_05.items():
                if col_f not in sub_df.columns: continue
                n_yes = int((sub_df[col_f]==1).sum())
                n_no  = int((sub_df[col_f]==0).sum())
                n_tot = n_yes + n_no
                pct   = int(n_yes/n_tot*100) if n_tot>0 else 0
                bar_w = max(2, pct)
                st.markdown(
                    f'<div style="padding:.35rem 0;border-bottom:1px solid #F5F3EF">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:.25rem">'
                    f'<span style="font-size:.83rem;font-weight:600;color:#1C1C1C">{lbl_f}</span>'
                    f'<span style="font-size:.78rem;color:#888">{n_yes:,} Yes · {n_no:,} No</span>'
                    f'</div>'
                    f'<div style="background:#F0EDE8;border-radius:3px;height:5px">'
                    f'<div style="width:{bar_w}%;height:5px;background:{RED};border-radius:3px"></div></div>'
                    f'<div style="font-size:.72rem;color:#AAA;margin-top:.15rem">{pct}% of assets have this feature</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        with af_col2:
            st.markdown('<div style="font-size:.72rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.6rem">Categorical features</div>', unsafe_allow_html=True)
            for col_f, lbl_f in CAT_LABEL_05.items():
                if col_f not in sub_df.columns: continue
                vc = sub_df[col_f].dropna()
                vc = vc[vc.astype(str).str.strip()!=""]
                if vc.empty: continue
                counts = vc.value_counts().head(6)
                values_html = "".join(
                    f'<span style="display:inline-block;background:#F5F3EF;border-radius:3px;'
                    f'padding:.15rem .5rem;margin:.1rem .2rem .1rem 0;font-size:.75rem;color:#3A3830">'
                    f'{str(v)[:30]} <span style="color:#AAA">({c:,})</span></span>'
                    for v,c in counts.items()
                )
                more2 = len(vc.value_counts()) - 6
                if more2 > 0:
                    values_html += f'<span style="font-size:.73rem;color:#AAA;margin-left:.2rem">+{more2} more</span>'
                st.markdown(
                    f'<div style="padding:.35rem 0;border-bottom:1px solid #F5F3EF;margin-bottom:.1rem">'
                    f'<div style="font-size:.83rem;font-weight:600;color:#1C1C1C;margin-bottom:.3rem">{lbl_f}</div>'
                    f'{values_html}</div>',
                    unsafe_allow_html=True
                )

    # ── Market impact comparison ──────────────────────────────────────────────
    with st.expander("Market impact — does a finding hold in every market?", expanded=False):
        mkt_base = df_full[df_full["_year"].isin(CTX["years"])] if CTX["years"] else df_full
        _applied = []
        for t,v in scope_filters:
            if t=="ou":    mkt_base=mkt_base[mkt_base["operating_unit_code"]==v]; _applied.append(f"OU {v}")
            if t=="brand": mkt_base=mkt_base[mkt_base["brand_name"]==v]; _applied.append(f"Brand {v}")
        if CTX["years"]: _applied.append(f"Year {', '.join(map(str, CTX['years']))}")
        st.markdown(f"""<div class="def-note">Uplift of the feature inside each market.
        <strong>Filters applied here:</strong> {', '.join(_applied) if _applied else 'none (all assets)'}.
        The Market and Campaign filters are ignored so that every market can be compared.
        Red = the feature worsens the metric in that market; ⚠ Reversal = significant and opposite to the overall result.</div>""",
        unsafe_allow_html=True)
        mkt_feat_opts = [(f, FEAT_LABEL.get(f,f)) for f in BIN_FEATS if f in sub_df.columns]
        mk1, mk2 = st.columns(2)
        with mk1:
            mkt_fi = st.selectbox("Feature", range(len(mkt_feat_opts)),
                                   format_func=lambda i: mkt_feat_opts[i][1], key="mkt_ins_feat")
        with mk2:
            mkt_mc_sel = st.selectbox("Metric", METRIC_NAMES, key="mkt_ins_mc")
        mkt_mc_col2 = mc_of(mkt_mc_sel)
        tired_note(mkt_mc_col2)
        mkt_feat_col, _ = mkt_feat_opts[mkt_fi]
        mkt_rows_ins = []
        for mkt in sorted(mkt_base["country_name"].dropna().unique()):
            mkt_sub2 = mkt_base[mkt_base["country_name"]==mkt]
            u2,sig2,n2 = compute_uplift(mkt_sub2, mkt_feat_col, mkt_mc_col2)
            if u2 is not None:
                mkt_rows_ins.append({"Market":mkt,"Uplift (pp)":round(u2,1),"Sig":sig2,"n":n2,
                                     "Reliability":rel_label(sig2,n2,int((~has_mask(mkt_sub2,mkt_feat_col)).sum()))})
        if mkt_rows_ins:
            mkt_ins_df = pd.DataFrame(mkt_rows_ins).sort_values("Uplift (pp)", ascending=lower_better(mkt_mc_col2))
            g_u2,_,_ = compute_uplift(mkt_base, mkt_feat_col, mkt_mc_col2)
            tbl_mkt_ins = '<table style="width:100%;border-collapse:collapse;font-size:.8rem"><thead><tr style="border-bottom:2px solid #EAE8E2"><th style="text-align:left;padding:.35rem .5rem;color:#888">Market</th><th style="text-align:center;color:#888">Uplift</th><th style="text-align:center;color:#7030A0">Sig</th><th style="text-align:center;color:#888">n</th><th style="text-align:center;color:#888">Reliability</th><th style="text-align:left;color:#888">Signal</th></tr></thead><tbody>'
            for i,(_, rm) in enumerate(mkt_ins_df.iterrows()):
                bg3 = "#FAFAF8" if i%2==0 else "#FFF"
                uv = rm["Uplift (pp)"]; uc3 = ucol(uv, mkt_mc_col2)
                sig3 = rm["Sig"]
                signal = ""
                if g_u2 is not None and np.sign(uv) != np.sign(g_u2) and sig3 in ("*","**","***"):
                    signal = '<span style="color:#C00020;font-size:.72rem">⚠ Reversal</span>'
                elif sig3 in ("**","***") and is_good(uv, mkt_mc_col2):
                    signal = '<span style="color:#2A8050;font-size:.72rem">✓ Works here</span>'
                tbl_mkt_ins += (f'<tr style="background:{bg3};border-bottom:1px solid #F5F3EF">'
                                f'<td style="padding:.28rem .5rem">{rm["Market"]}</td>'
                                f'<td style="text-align:center;font-weight:600;color:{uc3}">{uv:+.1f}pp</td>'
                                f'<td style="text-align:center;color:#7030A0">{sig3}</td>'
                                f'<td style="text-align:center;color:#AAA">{rm["n"]}</td>'
                                f'<td style="text-align:center">{rel_badge(rm["Reliability"])}</td>'
                                f'<td>{signal}</td></tr>')
            tbl_mkt_ins += '</tbody></table>'
            if g_u2 is not None:
                st.markdown(f'<div style="font-size:.78rem;color:#555;margin-bottom:.4rem">Overall (all markets above): <strong>{g_u2:+.1f}pp</strong></div>', unsafe_allow_html=True)
            st.markdown(tbl_mkt_ins, unsafe_allow_html=True)
            dl(mkt_ins_df.assign(feature=FEAT_LABEL.get(mkt_feat_col,mkt_feat_col), metric=mkt_mc_sel),
               "market_impact", "ic_mkt")
        else:
            st.markdown('<p style="color:#AAA;font-size:.8rem">Not enough market data.</p>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── All findings ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="sec-label">All findings ({len(fc):,})</div>', unsafe_allow_html=True)
    vmode=st.selectbox("View",["Table","Cards"], key="ic_view")
    sc2=["text","feature_display","metric_display","filter","filter_value",
         "evidence_uplift_pp","evidence_sig","reliability","evidence_n_has","coverage",
         "evidence_baseline_pp","quality_flags"]
    tbl_out=fc[[c for c in sc2 if c in fc.columns]].rename(columns={
        "text":"Insight","feature_display":"Feature","metric_display":"Metric","filter":"Scope",
        "filter_value":"Scope value","evidence_uplift_pp":"Uplift (pp)",
        "evidence_sig":"Sig.","reliability":"Reliability",
        "evidence_n_has":"n (with feature)","coverage":"Coverage","evidence_baseline_pp":"Baseline (pp)",
        "quality_flags":"Flags"}).reset_index(drop=True)
    dl(tbl_out, "insight_catalog_filtered", "ic_tbl")

    if vmode=="Table":
        st.dataframe(tbl_out, height=600, use_container_width=True)
    else:
        pg=max(1,int(np.ceil(len(fc)/20)))
        pnum=st.slider("Page",1,pg,1) if pg>1 else 1
        for _,r in fc.iloc[(pnum-1)*20:pnum*20].iterrows():
            v=r["evidence_uplift_pp"]; mcr=mc_of(r["metric_display"]); bc=ucol(v, mcr)
            flags=""
            if r.get("quality_flags",""):
                flags="".join(badge(x.strip(),"b-ns") for x in str(r["quality_flags"]).split(",") if x.strip())
            ou_snippet=""
            if has_ins and rulebook is not None:
                ou_het=rulebook[(rulebook["feature"]==r["feature"]) &
                                (rulebook["rule_type"]=="Heterogeneity") &
                                (rulebook["severity"]=="high")]
                if not ou_het.empty:
                    ou_snippet='<div style="margin-top:5px;font-size:.76rem;color:#906820">⚠ Reversal in some OUs — check Rulebook before briefing</div>'
            cb=feature_combinations(sub_df,r["feature"],mcr,top_n=1) if r["feature"] in sub_df.columns else []
            combo_snippet=""
            if cb:
                combo_snippet=(f'<div style="margin-top:5px;font-size:.76rem;color:#555">'
                               f'Best partner in your selection: <strong>{cb[0]["label"]}</strong> → '
                               f'{cb[0]["combined"]:+.1f}pp combined ({cb[0]["sig"]}, n={cb[0]["n"]})</div>')
            cov = r["coverage"]
            st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
              <div style="display:flex;justify-content:space-between">
                <div class="ic-feat">{r["feature_display"]}</div>
                <div>{badge(f'{"▲" if v>=0 else "▼"} {abs(v):.1f}pp', "b-pos" if r["_good"] else "b-neg")}</div>
              </div>
              <div class="ic-text">{r["text"]}</div>
              <div style="font-size:.72rem;color:#999;margin-top:3px">{FEAT_DESC.get(r["feature"],"")}</div>
              {ou_snippet}{combo_snippet}
              <div class="ic-meta">
                {rel_badge(r["reliability"])}
                {badge(r["metric_display"],"b-metric")}
                {sig_badge(r["evidence_sig"])}
                <span class="badge b-scope">n={int(r["evidence_n_has"]):,}{f" · {cov:.0%} of scope" if pd.notna(cov) else ""}</span>
                <span class="badge b-ns">base {r["evidence_baseline_pp"]:.1f}pp</span>
                {badge(r.get("filter","")+": "+str(r.get("filter_value","")),"b-scope")}
                {flags}
              </div>
            </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 06 — RULEBOOK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "06 · Rulebook":
    if not has_ins or rulebook is None:
        st.warning("rulebook.csv not found. Run the insight pipeline first."); st.stop()

    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">So What · Page 6 of 7</div>
      <div class="hero-title">Rulebook</div>
      <div class="hero-sub">Automated risk and opportunity alerts: trade-offs, market reversals, underused
        features and anti-patterns — before they end up in a brief.</div>
    </div>""", unsafe_allow_html=True)
    how_to("""<strong>Workflow:</strong> read the Key takeaways, then the <strong>high-severity</strong> alerts —
      these can invalidate a brief if missed. <strong>Heterogeneity</strong> is the most critical: a feature that works
      on average reverses in a specific OU, brand or market. <strong>Conflict</strong> flags metric trade-offs — brief with
      the campaign's main metric in mind. <strong>Opportunity</strong> points to features worth testing more.
      <br><br>By default you see alerts for your filters plus dataset-wide alerts; each card says which it is.
      The rulebook is precomputed per OU, brand and market — Year and Campaign filters do not change it.
      <br><br>""" + "<br>".join(f"<strong>{k}</strong> — {v}" for k, v in RULE_ACTION.items() if k != "Insight"))

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()

    RULE_BG={"Conflict":"#FFF0F0","Heterogeneity":"#FFFBF0","Boundary Condition":"#F8F0FF",
              "Opportunity":"#F0FFF4","Outlier":"#EBF3FB","Consensus":"#F0FFF4","Anti-pattern":"#FFF0F0"}
    RULE_AC={"Conflict":RED,"Heterogeneity":"#906820","Boundary Condition":"#7030A0",
              "Opportunity":"#1A7040","Outlier":"#1A3080","Consensus":"#1A7040","Anti-pattern":RED}

    rf1,rf2,rf3,rf4=st.columns(4)
    with rf1: rt_f=st.multiselect("Rule type",sorted(rulebook["rule_type"].unique()),
                                   default=sorted(rulebook["rule_type"].unique()))
    with rf2: sv_f=st.multiselect("Severity",["high","medium","low"],default=["high","medium"])
    with rf3: mt_f=st.multiselect("Metric",METRIC_NAMES,default=[])
    with rf4:
        mine=st.selectbox("Scope",["My filters + dataset-wide","All scopes"] if scope_filters else ["All scopes"], key="rb_scope")

    rb_f=rulebook.copy()
    if mine=="My filters + dataset-wide":
        m_=rb_f["scope"]=="Global"
        for t,v in scope_filters: m_ |= (rb_f["scope"]==SMAP.get(t,t)) & (rb_f["scope_value"]==v)
        rb_f=rb_f[m_]
    if rt_f: rb_f=rb_f[rb_f["rule_type"].isin(rt_f)]
    if sv_f: rb_f=rb_f[rb_f["severity"].isin(sv_f)]
    if mt_f: rb_f=rb_f[rb_f["metric"].apply(lambda m: any(rule_matches_metric(m, mc_of(x)) for x in mt_f))]
    rb_f=rb_f.assign(_mine=rb_f["scope"]!="Global").sort_values(
        ["_mine","severity"], ascending=[False,True], key=lambda c: c.map({"high":0,"medium":1,"low":2}) if c.name=="severity" else c)

    # ── Key takeaways ─────────────────────────────────────────────────────────
    hi=rb_f[rb_f["severity"]=="high"]
    tk=[f"<strong>{len(rb_f):,} alerts</strong> for "
        f"{scope_label if mine=='My filters + dataset-wide' else 'all scopes'} — {len(hi):,} high severity."]
    for rt in ["Heterogeneity","Anti-pattern","Conflict","Opportunity","Consensus"]:
        d=hi[hi["rule_type"]==rt] if rt not in ("Opportunity","Consensus") else rb_f[rb_f["rule_type"]==rt]
        if len(d):
            feats=", ".join(sorted({FEAT_LABEL.get(f,str(f).replace("_"," ").title()) for f in d["feature"]})[:4])
            tk.append(f"<strong>{rt}</strong> ({len(d)}): {feats} — {RULE_ACTION[rt]}")
    st.markdown('<div class="takeaway"><div class="takeaway-title">Key takeaways</div><ul>'
                + "".join(f"<li>{t}</li>" for t in tk) + "</ul></div>", unsafe_allow_html=True)

    rule_counts=rb_f["rule_type"].value_counts()
    if len(rule_counts):
        rcols=st.columns(min(len(rule_counts),7))
        for i,(rt,cnt) in enumerate(rule_counts.items()):
            bg=RULE_BG.get(rt,"#F9F9F9"); ac=RULE_AC.get(rt,MID)
            with rcols[i%7]:
                st.markdown(f"""<div style="background:{bg};border:1px solid rgba(0,0,0,.07);
                  border-left:3px solid {ac};border-radius:6px;padding:.85rem .95rem;
                  margin-bottom:.7rem;text-align:center">
                  <div style="font-size:1.4rem;font-weight:700;color:{ac}">{cnt}</div>
                  <div style="font-size:.66rem;font-weight:700;color:{ac};
                    text-transform:uppercase;letter-spacing:.06em">{rt}</div>
                </div>""",unsafe_allow_html=True)

    st.markdown(f"<span style='color:{MID};font-size:.83rem'>{len(rb_f)} entries</span>",unsafe_allow_html=True)
    dl(rb_f.drop(columns=["_mine"]).assign(action=rb_f["rule_type"].map(RULE_ACTION)), "rulebook_filtered", "rb")

    SEV_COL={"high":RED,"medium":"#906820","low":"#2A8050"}
    _show=st.selectbox("Alerts to show", ["25","50","100","All"], key="rb_show")
    show_n=len(rb_f) if _show=="All" else int(_show)
    for _,r in rb_f.head(show_n).iterrows():
        bg=RULE_BG.get(r["rule_type"],"#F9F9F9"); ac=RULE_AC.get(r["rule_type"],MID)
        sc2=SEV_COL.get(r.get("severity","medium"),MID)
        st.markdown(f"""<div style="background:{bg};border:1px solid rgba(0,0,0,.07);
          border-left:4px solid {ac};border-radius:6px;padding:11px 15px;margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div style="font-size:.67rem;font-weight:700;text-transform:uppercase;
              letter-spacing:.08em;color:{ac};margin-bottom:4px">{r["rule_type"]}</div>
            <span style="font-size:.67rem;font-weight:700;color:{sc2};
              text-transform:uppercase">{r.get("severity","").upper()}</span>
          </div>
          <div style="font-size:.83rem;color:#333;line-height:1.5">{r["text"]}</div>
          <div class="iw-action">→ {RULE_ACTION.get(r["rule_type"],"")}</div>
          <div style="display:flex;gap:5px;margin-top:.4rem;flex-wrap:wrap">
            {badge(FEAT_LABEL.get(r.get("feature",""), r.get("feature","")),"b-metric")}
            {badge(_scope_txt(r.get("scope",""), r.get("scope_value","")),"b-scope")}
          </div>
        </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 07 — CAMPAIGN COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "07 · Campaign Comparison":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">So What · Page 7 of 7</div>
      <div class="hero-title">Campaign Comparison</div>
      <div class="hero-sub">One row per campaign, aggregated across all its markets and assets —
        for campaign reports and campaign-vs-campaign comparisons.</div>
    </div>""", unsafe_allow_html=True)
    how_to("""<strong>Columns:</strong><br>
      <strong>Score</strong> — average of all the campaign's assets in your selection (each asset counts once).<br>
      <strong>Market-balanced score</strong> — average of the campaign's per-market averages, so a market with many
      assets does not dominate.<br>
      <strong>vs average</strong> — Score minus the benchmark chosen in <em>Compare against</em>.<br>
      <strong>vs local market avg</strong> — each asset minus the average of all benchmark assets in its own market,
      then averaged. This removes differences in market baselines and is the fairest way to compare campaigns that
      ran in different markets.<br><br>
      Different markets of one campaign can have very different assets, so check the market split (expander) before
      quoting a single campaign number.""")

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()
    bench_df, bench_label = CTX["bench_df"], CTX["bench_label"]

    cm_lbl = st.selectbox("Metric", METRIC_NAMES, key="cc_mc")
    cm = mc_of(cm_lbl)
    tired_note(cm)

    d = sub_df[sub_df[cm].notna()].copy()
    loc_avg = bench_df.groupby("country_name")[cm].mean()
    d["_vs_local"] = (d[cm] - d["country_name"].map(loc_avg)) * 100
    bench_mean = bench_df[cm].dropna().mean() * 100
    mkt_means = d.groupby(["campaign_sk_id","country_name"])[cm].mean().groupby(level=0).mean()*100
    agg = (d.groupby(["campaign_sk_id","campaign_display_name","campaign_code"])
             .agg(markets=("country_name", lambda x: ", ".join(sorted(x.dropna().unique()))),
                  n_markets=("country_name","nunique"), n_assets=(cm,"count"),
                  years=("_year", lambda x: ", ".join(str(int(y)) for y in sorted(x.dropna().unique()))),
                  score=(cm, "mean"), vs_local=("_vs_local","mean"))
             .reset_index())
    agg["score"] *= 100
    agg["balanced"] = agg["campaign_sk_id"].map(mkt_means)
    agg["vs_avg"] = agg["score"] - bench_mean
    agg = agg.sort_values("vs_local", ascending=lower_better(cm))

    st.markdown(f'<div class="def-note">{len(agg):,} campaigns · benchmark = {bench_label} ({bench_mean:.1f}pp). '
                f'Sorted by <strong>vs local market avg</strong>. Campaigns with fewer than 5 assets are directional.</div>',
                unsafe_allow_html=True)
    tbl = ('<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.82rem"><thead>'
           '<tr style="border-bottom:2px solid #EAE8E2;color:#888;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em">'
           '<th style="text-align:left;padding:.4rem .5rem">Campaign</th><th style="text-align:left">Markets</th>'
           '<th>Assets</th><th>Score</th><th>Market-balanced</th><th>vs average</th><th>vs local market avg</th></tr></thead><tbody>')
    for i, (_, r) in enumerate(agg.iterrows()):
        bg = "#FAFAF8" if i % 2 == 0 else "#FFF"
        low = ' <span style="color:#906820">⚠</span>' if r["n_assets"] < 5 else ""
        tbl += (f'<tr style="background:{bg};border-bottom:1px solid #F5F3EF">'
                f'<td style="padding:.35rem .5rem"><strong>{str(r["campaign_display_name"])[:45]}</strong>'
                f'<div style="font-size:.7rem;color:#AAA">{r["campaign_code"]} · {r["years"]}</div></td>'
                f'<td style="color:#555;max-width:220px">{short_list(r["markets"], 4)}</td>'
                f'<td style="text-align:center;color:#888">{int(r["n_assets"])}{low}</td>'
                f'<td style="text-align:center;font-weight:600">{r["score"]:.1f}</td>'
                f'<td style="text-align:center">{r["balanced"]:.1f}</td>'
                f'<td style="text-align:center;color:{ucol(r["vs_avg"], cm, .5)}">{r["vs_avg"]:+.1f}</td>'
                f'<td style="text-align:center;font-weight:700;color:{ucol(r["vs_local"], cm, .5)}">{r["vs_local"]:+.1f}</td></tr>')
    tbl += "</tbody></table></div>"
    st.markdown(tbl, unsafe_allow_html=True)
    dl(agg.drop(columns=["campaign_sk_id"]).rename(columns={
        "campaign_display_name":"Campaign","campaign_code":"Code","markets":"Markets","n_markets":"# markets",
        "n_assets":"# assets","years":"Years","score":f"{cm_lbl} score (pp)","balanced":"Market-balanced score (pp)",
        "vs_avg":"vs average (pp)","vs_local":"vs local market avg (pp)"}).assign(Benchmark=bench_label).round(2),
       f"campaign_comparison_{cm_lbl}", "cc")

    # ── side-by-side across all metrics ───────────────────────────────────────
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Compare campaigns across all metrics</div>', unsafe_allow_html=True)
    _ids = agg["campaign_sk_id"].tolist()
    _nm = dict(zip(agg["campaign_sk_id"], agg["campaign_display_name"].astype(str) + " (" + agg["campaign_code"].astype(str) + ")"))
    pick = st.multiselect("Campaigns", _ids, default=_ids[:min(3, len(_ids))], format_func=lambda i: _nm[i], key="cc_pick")
    if pick:
        rows = []
        for mc2 in ALL_METRICS:
            if mc2 not in sub_df.columns: continue
            la = bench_df.groupby("country_name")[mc2].mean()
            row = {"Metric": M_LABEL.get(mc2, mc2) + (" (↓ better)" if lower_better(mc2) else "")}
            for cid in pick:
                g = sub_df[(sub_df["campaign_sk_id"]==cid) & sub_df[mc2].notna()]
                if g.empty: row[_nm[cid]] = "—"; continue
                v = g[mc2].mean()*100; vl = ((g[mc2]-g["country_name"].map(la))*100).mean()
                row[_nm[cid]] = f"{v:.1f} ({vl:+.1f} vs local)"
            rows.append(row)
        side = pd.DataFrame(rows)
        st.dataframe(side, hide_index=True, use_container_width=True)
        dl(side, "campaigns_side_by_side", "cc_side")

        with st.expander("Market split for the selected campaigns"):
            ms = (sub_df[sub_df["campaign_sk_id"].isin(pick) & sub_df[cm].notna()]
                  .groupby(["campaign_display_name","country_name"])[cm].agg(["mean","count"]).reset_index())
            ms["mean"] *= 100
            ms["local market avg"] = ms["country_name"].map(loc_avg)*100
            ms["vs local (pp)"] = ms["mean"] - ms["local market avg"]
            ms = ms.rename(columns={"campaign_display_name":"Campaign","country_name":"Market",
                                    "mean":f"{cm_lbl} (pp)","count":"n assets"}).round(2)
            st.dataframe(ms, hide_index=True, use_container_width=True)
            dl(ms, "campaign_market_split", "cc_ms")


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>Uplift = mean(feature present) &minus; mean(feature absent) &nbsp;&middot;&nbsp;
    Significance: Mann-Whitney U &nbsp;&middot;&nbsp; pp = percentage points &nbsp;&middot;&nbsp;
    SCD = 10% See + 30% Connect + 60% Do &nbsp;&middot;&nbsp; Tiredness: lower is better</span>
  <span>The Coca&#8209;Cola Company &nbsp;&middot;&nbsp; Asset Intelligence</span>
</div>
""", unsafe_allow_html=True)
