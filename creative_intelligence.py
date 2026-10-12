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
    initial_sidebar_state="expanded",
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
    "additional_elements_and_product_placement": "Secondary visual elements and product placement style in scene.",
    "most_frequently_used_word_in_creative": "The dominant text/copy message. Affects comprehension and key message recall.",
}
SCOPE_COL = {"ou":"operating_unit_code","category":"category",
             "brand":"brand_name","market":"country_name"}  # market kept in SCOPE_COL for backward compat with alerts
ALL_METRICS = list(dict.fromkeys(list(METRICS.keys()) + NEW_METRICS))


# ── html helpers ──────────────────────────────────────────────────────────────
def badge(txt, cls): return f'<span class="badge {cls}">{txt}</span>'
def bpos(v):
    if pd.isna(v): return badge("—","b-ns")
    cls = "b-pos" if v>=0 else "b-neg"
    return badge(f'{"▲" if v>=0 else "▼"} {abs(v):.1f}pp', cls)
def sig_badge(s):
    if s in ("***","**","*"): return badge(s,"b-sig")
    return badge("ns","b-ns")
def conf_badge(c):
    ranges = {"high": "80-100%", "medium": "60-79%", "low": "<60%"}
    label = c.upper() + (f" ({ranges[c]})" if c in ranges else "")
    return badge(label, f"b-{c[:2]}")

def render_alerts(items, max_items=6):
    if not items: return
    import re
    def highlight_pp(text):
        # highlight patterns like +5.1pp, -4.4pp, 5.1pp with bold colored spans
        def repl(m):
            val = m.group(0)
            color = "#2A8050" if "+" in val or (val[0].isdigit()) else "#C00020"
            # detect sign more carefully
            try:
                num = float(re.sub(r"[^\d.\-]","", val.replace("pp","")))
                color = "#2A8050" if num >= 0 else "#C00020"
            except: pass
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
        # "all 3 metrics" → "all metrics (Attention, Persuasion, Likeability)"
        text = re.sub(r'all 3 metrics', 'all metrics (Attention, Persuasion, Likeability)', text)
        return text
    html = '<div class="insight-strip">'
    for it in items[:max_items]:
        css, icon = RULE_STYLE.get(it["type"], ("warn-insight","ℹ"))
        display_text = expand_metric_abbrevs(highlight_pp(it["text"]))
        html += (f'<div class="insight-warn {css}">'
                 f'<div class="iw-icon">{icon}</div>'
                 f'<div class="iw-body">'
                 f'<div class="iw-type">{it["type"]}</div>'
                 f'<div style="font-size:.82rem;color:#333">{display_text}</div>'
                 f'</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def get_scope_alerts(scope_filters, max_items=6):
    if not has_ins or rulebook is None: return []
    smap = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
    if not scope_filters:
        rb = rulebook[(rulebook["scope"]=="Global") &
                      (rulebook["rule_type"].isin(["Conflict","Anti-pattern","Consensus"]))]
    else:
        mask = pd.Series(False, index=rulebook.index)
        for t,v in scope_filters:
            mask |= ((rulebook["scope"]==smap.get(t,t)) & (rulebook["scope_value"]==v))
        mask |= (rulebook["scope"]=="Global")
        rb = rulebook[mask]
    rb = rb.sort_values("severity", key=lambda s: s.map({"high":0,"medium":1,"low":2}))
    return [{"type":r["rule_type"],"text":r["text"]} for _,r in rb.head(max_items).iterrows()]

def get_feature_alerts(feature, scope_filters, metric_col, max_items=3):
    if not has_ins or rulebook is None or catalog is None: return []
    smap = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
    ml_map = {"Attention_T2B":"Attention","Persuasion_T2B":"Persuasion",
              "Likeability_Love_Like_T2B":"Likeability","SCD_score":"SCD Score",
              "Experience_Recall_T2B":"Experience Recall","Brand_Linkage_T2B":"Brand Linkage",
              "Uniqueness_T2B":"Uniqueness","Shareability_T2B":"Shareability"}
    ml = ml_map.get(metric_col, metric_col)
    items = []
    # rulebook for this feature
    rb = rulebook[rulebook["feature"]==feature]
    if scope_filters:
        sv = {smap.get(t,t):v for t,v in scope_filters}
        rb_s = rb[rb.apply(lambda r: r["scope"] in sv and sv.get(r["scope"])==r["scope_value"],axis=1)]
        rb = pd.concat([rb_s, rb[rb["scope"]=="Global"]]).drop_duplicates()
    else:
        rb = rb[rb["scope"]=="Global"]
    rb = rb.sort_values("severity", key=lambda s: s.map({"high":0,"medium":1,"low":2}))
    for _,r in rb.head(max_items).iterrows():
        items.append({"type":r["rule_type"],"text":r["text"]})
    # catalog insight
    if len(items) < max_items:
        cat = catalog[(catalog["feature"]==feature) &
                      (catalog["metric_display"]==ml) &
                      (catalog["confidence"].isin(["high","medium"]))]
        added = False
        if scope_filters:
            for t,v in scope_filters:
                sl = smap.get(t,t)
                for _,r in cat[(cat["filter"]==sl)&(cat["filter_value"]==v)].head(1).iterrows():
                    items.append({"type":"Insight","text":r["text"]}); added=True
        if not added:
            for _,r in cat[cat["filter"]=="Global"].head(1).iterrows():
                items.append({"type":"Insight","text":r["text"]})
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

def compute_uplift_per_value(sub, feat, mc, min_n=5):
    """For multi-value comma-separated features (occasions, passion_point),
    compute uplift per individual value vs assets that have NO value for this feature."""
    if feat not in sub.columns or mc not in sub.columns: return []
    from collections import Counter
    absent = sub.loc[~(sub[feat].notna() & (sub[feat].astype(str).str.strip()!="")), mc].dropna()
    if len(absent) < min_n: return []
    absent_mean = absent.mean()
    # Count all individual values
    val_counter = Counter()
    for row_val in sub[feat].dropna():
        for part in str(row_val).split(","):
            part = part.strip()
            if part: val_counter[part] += 1
    rows = []
    for val, cnt in val_counter.most_common():
        mask = sub[feat].astype(str).str.contains(val, regex=False, na=False)
        g = sub.loc[mask, mc].dropna()
        if len(g) < min_n: continue
        u = (g.mean() - absent_mean) * 100
        try:
            from scipy.stats import mannwhitneyu as mwu
            _, p = mwu(g, absent, alternative="two-sided")
        except: p = 1.0
        sig = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))
        rows.append({"value": val, "uplift": round(u,2), "sig": sig, "n": len(g), "n_all": cnt})
    rows.sort(key=lambda x: abs(x["uplift"]), reverse=True)
    return rows[:15]

def feature_combinations(sub, target, mc, top_n=5):
    """Top binary feature partners for target feat on metric mc."""
    if target not in sub.columns or mc not in sub.columns: return []
    gm = sub[mc].dropna().mean()*100
    g1t = sub.loc[sub[target]==1,mc].dropna() if target in BIN_FEATS \
          else sub.loc[sub[target].notna()&(sub[target].astype(str).str.strip()!=""),mc].dropna()
    if len(g1t)<3: return []
    solo_t = g1t.mean()*100 - gm
    rows=[]
    for f in BIN_FEATS:
        if f==target or f not in sub.columns: continue
        if target in BIN_FEATS:
            mb=(sub[target]==1)&(sub[f]==1)
        else:
            mt=sub[target].notna()&(sub[target].astype(str).str.strip()!="")
            mb=mt&(sub[f]==1)
        nb=int(mb.sum())
        if nb<3: continue
        cu=sub.loc[mb,mc].dropna().mean()*100-gm
        g1p=sub.loc[sub[f]==1,mc].dropna()
        sp=g1p.mean()*100-gm if len(g1p)>=3 else 0
        rows.append({"partner":f,"label":FEAT_LABEL.get(f,f),
                     "combined":round(cu,2),"solo_t":round(solo_t,2),
                     "solo_p":round(sp,2),"synergy":round(cu-max(solo_t,sp),2),"n":nb})
    rows.sort(key=lambda x:x["combined"],reverse=True)
    return rows[:top_n]


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
            n_row=row.get("n",0)
            if n_row<3: continue
            raws.append(str(row["val"]))
            up=row.get("uplift_pp")
            if up is not None and not row.get("is_baseline",False):
                disp.append(f"{row['label']}  ({'+'if up>=0 else ''}{up:.1f}pp, n={n_row:,})")
            else:
                disp.append(f"{row['label']}  (n={n_row:,})")
    elif feat in BIN_FEATS and feat in sub_df.columns:
        n1=int((sub_df[feat]==1).sum()); n0=int((sub_df[feat]==0).sum())
        if n1>=3: raws+=["1"]; disp+=[f"Yes  (n={n1:,})"]
        if n0>=3: raws+=["0"]; disp+=[f"No  (n={n0:,})"]
    elif feat in sub_df.columns:
        for v in sorted(sub_df[feat].dropna().unique()):
            n=int((sub_df[feat]==v).sum())
            if n>=3: raws.append(str(v)); disp.append(f"{str(v)[:32]}  (n={n:,})")
    return raws,disp

def bar_html(frac):
    pct=max(2,min(100,frac*100))
    return (f'<div style="background:#F0EDE8;border-radius:2px;height:5px;">'
            f'<div style="width:{pct:.0f}%;height:5px;background:#E8002D;border-radius:2px;"></div></div>')

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
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">The Coca&#8209;Cola Company</div>
    <div class="topbar-pipe"></div>
    <div class="topbar-sub">Asset Intelligence</div>
  </div>
  <div class="topbar-sub">Asset Feature Explorer</div>
</div>
""", unsafe_allow_html=True)

# ── navigation ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "ℹ · About this App",
        "01 · Overview & Performance",
        "02 · Feature Impact",
        "03 · Combination Explorer",
        "04 · Feature Combinations & OU Impact",
        "05 · Insight Catalog",
        "06 · Rulebook",
    ])
    st.markdown("---")
    st.markdown(f"<small style='color:#888'>{len(df_full):,} assets · {len(results)} precomputed scopes</small>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FILTER BAR — rendered per-page so hero appears above filters
# ══════════════════════════════════════════════════════════════════════════════
def render_filters():
    """Render the filter bar and return (sub_df, scope_filters, scope_key, min_n, scope_label).
    Call this inside each page block, after the hero."""
    st.markdown('<div class="sec-label" style="margin-top:1.4rem">Filters</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        ou_opts = ["All OUs"] + sorted(df_full["operating_unit_code"].dropna().unique().tolist())
        sel_ou = st.selectbox("Operating Unit", ou_opts, key="f_ou")
    with fc2:
        brand_opts = ["All Brands"] + sorted(df_full["brand_name"].dropna().unique().tolist())
        sel_brand = st.selectbox("Brand", brand_opts, key="f_brand")
    with fc3:
        _csub = df_full.copy()
        if sel_ou != "All OUs": _csub = _csub[_csub["operating_unit_code"] == sel_ou]
        if sel_brand != "All Brands": _csub = _csub[_csub["brand_name"] == sel_brand]
        _cdf = (_csub[["campaign_sk_id","campaign_display_name","campaign_code"]]
                .drop_duplicates("campaign_sk_id").sort_values("campaign_display_name"))
        camp_opts = ["All Campaigns"] + [f"{r.campaign_display_name} ({r.campaign_code})"
                                          for r in _cdf.itertuples()]
        camp_ids  = [None] + _cdf["campaign_sk_id"].tolist()
        sel_camp_i = st.selectbox("Campaign", range(len(camp_opts)),
                                   format_func=lambda i: camp_opts[i], key="f_camp")
        sel_camp = camp_ids[sel_camp_i]
    with fc4:
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
    if sel_camp is not None:
        sub_df = sub_df[sub_df["campaign_sk_id"] == sel_camp]
    if sel_mkts:
        sub_df = sub_df[sub_df["country_name"].isin(sel_mkts)]
        for m in sel_mkts:
            scope_filters.append(("market", m))

    scope_key = get_scope_key(scope_filters[:1]) if sel_camp is None else None
    min_n = 5 if scope_filters else 10

    chip_parts = []
    if sel_ou != "All OUs":    chip_parts.append(f"OU: {sel_ou}")
    if sel_brand != "All Brands": chip_parts.append(f"Brand: {sel_brand}")
    if sel_camp is not None:   chip_parts.append(f"Campaign: {camp_opts[sel_camp_i].split('(')[0].strip()}")
    if sel_mkts:               chip_parts.append(f"Markets: {', '.join(sel_mkts[:2])}{'...' if len(sel_mkts)>2 else ''}")
    if chip_parts:
        chips = "".join(f'<span class="scope-chip">{p}</span>' for p in chip_parts)
        st.markdown(f'<div style="margin:.4rem 0 0 0">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:.74rem;color:#CCC8C2;margin:.4rem 0 0 0">Average · All brands · All campaigns · All markets</p>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    scope_label = "Global" if not chip_parts else " · ".join(chip_parts)
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
        Use the <strong>Filters</strong> at the top to scope results by OU, Brand, Campaign, or Market.
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
         "Red bars in the OU chart = the feature reverses in that OU.",
         "Critical step before briefing. A feature that works at +5pp globally "
         "may be −3pp in a specific OU. Always check Part B before writing a global brief."),
        ("05 · Insight Catalog", "sowhat", "SO WHAT",
         "The validated evidence base: every statistically tested finding from the full pipeline, "
         "filterable by scope, metric, confidence level, and direction. Includes a market impact "
         "section to see if findings hold across markets. Feature distributions shown for context.",
         "High-confidence findings are safe to use in briefs directly. "
         "Medium = directional only. Low = do not brief. "
         "Filter by OU or Brand to see scope-specific findings."),
        ("06 · Rulebook", "sowhat", "SO WHAT",
         "Automated alerts: Conflicts (feature improves one metric, hurts another), "
         "Heterogeneity (works on average but reverses in a specific scope), "
         "Opportunities (positive feature used in <20% of assets), "
         "Anti-patterns (negative across all metrics in a scope).",
         "Always review <strong>high-severity</strong> entries before writing any brief. "
         "Heterogeneity alerts are the most important — they prevent you from "
         "briefing something globally that will hurt specific markets."),
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
        ("Tiredness_T2B",             "Tiredness",          "Are people tired of seeing this ad? ⚠ Lower is better. High tiredness = audience fatigue → rotate creative."),
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
        ("Confidence (High/Medium/Low)", "High (80-100%): significant at ** or ***, 50+ assets per group — safe to brief. "
         "Medium (60-79%): significant but smaller sample — directional only. "
         "Low (<60%): not significant or tiny sample — do not brief."),
        ("Baseline", "The average score of assets that do NOT have the feature. "
         "Uplift is measured relative to this starting point."),
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
      <div class="hero-eyebrow">What Happened · Page 1 of 6</div>
      <div class="hero-title">Overview &amp; Performance</div>
      <div class="hero-sub">Your starting point. Metric scorecards for all 8 KPIs,
        a data-driven insight summary, campaign breakdown, and top assets ranked by any metric.
        Use this page to understand <em>what the numbers say</em> before digging into causes.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#EBF3FB;border-left:3px solid #2D5BE3;border-radius:0 4px 4px 0;
      padding:.7rem 1rem;font-size:.82rem;color:#1A3A6A;margin-bottom:1.2rem;line-height:1.65;">
      <strong>Analyst workflow:</strong> Start with the Insight Summary to get the headline read.
      Then check the Campaign breakdown to see if performance differs across campaigns.
      Use the Asset Viewer to inspect individual assets that stand out.
      Once you know what happened, move to <strong>02 · Feature Impact</strong> to find out why.
    </div>""", unsafe_allow_html=True)

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()


    # KPI cards — row 1: assets + core 3
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-val">{len(sub_df):,}</div>
          <div class="kpi-lbl">Assets</div>
          <div class="kpi-sub">{sub_df["campaign_sk_id"].nunique():,} campaigns</div>
        </div>""", unsafe_allow_html=True)
    for ml,(mc,col) in zip(
        ["Attention","Persuasion","Likeability"],
        [("Attention_T2B",k2),("Persuasion_T2B",k3),("Likeability_Love_Like_T2B",k4)]
    ):
        if mc in sub_df.columns:
            mv=sub_df[mc].dropna().mean()*100
            with col:
                st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {RED}">
                  <div class="kpi-val">{mv:.1f}<span style="font-size:1.1rem">pp</span></div>
                  <div class="kpi-lbl">{ml}</div>
                  <div class="kpi-sub">Mean score in scope</div>
                </div>""", unsafe_allow_html=True)

    # KPI cards — row 2: additional metrics
    n1,n2,n3,n4,n5 = st.columns(5)
    for ml,mc,col in [
        ("Experience Recall","Experience_Recall_T2B",n1),
        ("Brand Linkage",    "Brand_Linkage_T2B",    n2),
        ("Uniqueness",       "Uniqueness_T2B",        n3),
        ("Shareability",     "Shareability_T2B",      n4),
        ("Tiredness",        "Tiredness_T2B",         n5),
    ]:
        if mc in sub_df.columns:
            mv=sub_df[mc].dropna().mean()*100
            with col:
                st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {RED}">
                  <div class="kpi-val">{mv:.1f}<span style="font-size:1.1rem">pp</span></div>
                  <div class="kpi-lbl">{ml}</div>
                  <div class="kpi-sub">Mean score in scope</div>
                </div>""", unsafe_allow_html=True)

    # Glossary
    with st.expander("What do these numbers mean?"):
        st.markdown("""
**pp (percentage points)** — scores are expressed as the percentage of survey respondents who gave a positive answer.
A score of 62pp means 62% of people who watched the ad responded positively to that question.

**Attention** — did the ad successfully grab people's attention? (Top 2 Box survey response)

**Persuasion** — did the ad make people more likely to buy the product? (Top 2 Box)

**Likeability** — did people like the ad — did they love it or like it? (Love + Like combined)

**Uplift** — the difference in average score between ads that have a feature and ads that don't. An uplift of +3pp means ads with that feature score 3 percentage points higher on average.
        """)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── CHANGE 13: Insight summary ────────────────────────────────────────────
    st.markdown('<div class="sec-label">Insight summary</div>', unsafe_allow_html=True)
    with st.expander("View performance summary across all metrics", expanded=True):
        # Compute per-metric means
        metric_summary = {}
        for mc_s, ml_s in M_LABEL.items():
            if mc_s in sub_df.columns:
                v = sub_df[mc_s].dropna()
                metric_summary[ml_s] = round(v.mean()*100, 1) if len(v)>0 else None

        # Global means for comparison
        global_means = {}
        for mc_s, ml_s in M_LABEL.items():
            if mc_s in df_full.columns:
                v = df_full[mc_s].dropna()
                global_means[ml_s] = round(v.mean()*100, 1) if len(v)>0 else None

        # Build metric performance rows
        perf_html = '<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1rem">'
        for ml_s, val_s in metric_summary.items():
            if val_s is None: continue
            gv = global_means.get(ml_s)
            diff = round(val_s - gv, 1) if gv else 0
            is_tiredness = ml_s == "Tiredness"
            # For Tiredness lower is better
            if is_tiredness:
                diff_col = GREEN if diff < -0.5 else (RED if diff > 0.5 else AMBER)
            else:
                diff_col = GREEN if diff > 0.5 else (RED if diff < -0.5 else AMBER)
            sign = "+" if diff >= 0 else ""
            border_col = RED if (is_tiredness and diff > 0.5) else (GREEN if diff > 0.5 else BORDER)
            perf_html += (
                f'<div style="background:#fff;border:1px solid {border_col};border-radius:6px;padding:.55rem .9rem;min-width:120px">' +
                f'<div style="font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#888">{ml_s}</div>' +
                f'<div style="font-size:1.35rem;font-weight:700;color:{DARK};font-family:Merriweather,serif">{val_s:.1f}pp</div>' +
                f'<div style="font-size:.72rem;color:{diff_col}">{sign}{diff:.1f}pp vs average</div>' +
                f'</div>'
            )
        perf_html += '</div>'
        st.markdown(perf_html, unsafe_allow_html=True)

        # Narrative
        shr_v = metric_summary.get("Shareability"); shr_g = global_means.get("Shareability")
        rec_v = metric_summary.get("Experience Recall"); rec_g = global_means.get("Experience Recall")
        trd_v = metric_summary.get("Tiredness"); trd_g = global_means.get("Tiredness")
        att_v = metric_summary.get("Attention"); att_g = global_means.get("Attention")
        pers_v = metric_summary.get("Persuasion")

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
                narratives.append(f"⚠ **Tiredness** is above average at **{trd_v:.1f}pp** (+{diff_trd:.1f}pp) — audience fatigue risk. Consider rotating creative or reducing frequency.")
            elif diff_trd < -0.5:
                narratives.append(f"✓ **Tiredness** is below average at **{trd_v:.1f}pp** ({diff_trd:.1f}pp) — audience is not fatigued.")

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
            camp_mc_sel = st.selectbox("Metric", ["Attention","Persuasion","Likeability","Experience Recall","Brand Linkage","Uniqueness","Shareability","Tiredness"], key="camp_mc")
            camp_mc_col = {"Attention":"Attention_T2B","Persuasion":"Persuasion_T2B","Likeability":"Likeability_Love_Like_T2B","Experience Recall":"Experience_Recall_T2B","Brand Linkage":"Brand_Linkage_T2B","Uniqueness":"Uniqueness_T2B","Shareability":"Shareability_T2B","Tiredness":"Tiredness_T2B"}[camp_mc_sel]
            camp_grp = (sub_df.groupby(["campaign_sk_id","campaign_display_name"])[camp_mc_col]
                        .agg(mean_val="mean", n="count").reset_index())
            camp_grp["mean_val"] = camp_grp["mean_val"] * 100
            camp_grp = camp_grp.sort_values("mean_val", ascending=False)
            global_mean_camp = df_full[camp_mc_col].dropna().mean()*100 if camp_mc_col in df_full.columns else None
            tbl_camp = '<table style="width:100%;border-collapse:collapse;font-size:.82rem"><thead><tr style="border-bottom:2px solid #EAE8E2"><th style="text-align:left;padding:.4rem .6rem;color:#888">Campaign</th><th style="text-align:center;color:#888">Score</th><th style="text-align:center;color:#888">vs Average</th><th style="text-align:center;color:#888">n assets</th></tr></thead><tbody>'
            for i, (_, rc) in enumerate(camp_grp.iterrows()):
                bg_c = "#FAFAF8" if i%2==0 else "#FFF"
                diff_c = rc["mean_val"] - global_mean_camp if global_mean_camp else 0
                dc = GREEN if diff_c > 0.5 else (RED if diff_c < -0.5 else AMBER)
                sign_c = "+" if diff_c >= 0 else ""
                tbl_camp += (f'<tr style="background:{bg_c};border-bottom:1px solid #F5F3EF">'
                             f'<td style="padding:.35rem .6rem">{str(rc["campaign_display_name"])[:50]}</td>'
                             f'<td style="text-align:center;font-weight:600">{rc["mean_val"]:.1f}pp</td>'
                             f'<td style="text-align:center;color:{dc}">{sign_c}{diff_c:.1f}pp</td>'
                             f'<td style="text-align:center;color:#AAA">{int(rc["n"])}</td></tr>')
            tbl_camp += '</tbody></table>'
            st.markdown(tbl_camp, unsafe_allow_html=True)
            # Top features per campaign (top 3 campaigns)
            st.markdown('<div style="margin-top:1rem;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Top feature per campaign (most significant uplift)</div>', unsafe_allow_html=True)
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
                    col_c = GREEN if best_u >= 0 else RED
                    st.markdown(f'<div style="font-size:.8rem;padding:.2rem 0;color:#333"><strong>{str(rc["campaign_display_name"])[:40]}</strong>: {FEAT_LABEL.get(best_feat,best_feat)} <span style="color:{col_c}">{"+" if best_u>=0 else ""}{best_u:.1f}pp</span> ({best_sig})</div>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── Performance findings summary ──────────────────────────────────────────
    st.markdown('<div class="sec-label">Performance findings</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    A summary of which creative features drove positive or negative performance,
    and how each market performed relative to the average.
    Use this to quickly identify what to repeat and what to avoid in the next brief.</div>""",
    unsafe_allow_html=True)

    _fs_col1, _fs_col2 = st.columns(2)

    with _fs_col1:
        # Feature drivers summary
        st.markdown('<div style="font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6A6660;margin-bottom:.6rem">Feature drivers</div>', unsafe_allow_html=True)
        pos_feats, neg_feats = [], []
        for _feat in ALL_FEATS:
            if _feat not in sub_df.columns: continue
            for _mc_key in ["Attention_T2B","Persuasion_T2B","Likeability_Love_Like_T2B",
                             "Shareability_T2B","Experience_Recall_T2B","Tiredness_T2B"]:
                if _mc_key not in sub_df.columns: continue
                # Lower min_n to 2 so small campaign selections still show results
                if _feat in BIN_FEATS:
                    _g1 = sub_df.loc[sub_df[_feat]==1, _mc_key].dropna()
                    _g0 = sub_df.loc[sub_df[_feat]==0, _mc_key].dropna()
                else:
                    _mask = sub_df[_feat].notna() & (sub_df[_feat].astype(str).str.strip()!="")
                    _g1 = sub_df.loc[_mask, _mc_key].dropna()
                    _g0 = sub_df.loc[~_mask, _mc_key].dropna()
                if len(_g1) < 2 or len(_g0) < 2: continue
                _u = (_g1.mean() - _g0.mean()) * 100
                try:
                    from scipy.stats import mannwhitneyu as _mwu
                    _, _p = _mwu(_g1, _g0, alternative="two-sided")
                except: _p = 1.0
                _sig = "***" if _p<.001 else ("**" if _p<.01 else ("*" if _p<.05 else "ns"))
                _lbl = FEAT_LABEL.get(_feat, _feat)
                _mlbl = M_LABEL.get(_mc_key, _mc_key)
                entry = (_lbl, _mlbl, round(_u,2), _sig, len(_g1))
                if _u > 0:
                    pos_feats.append(entry)
                else:
                    neg_feats.append(entry)

        def _dedup(rows):
            seen = {}
            for lbl, mlbl, u, sig, n in sorted(rows, key=lambda x: abs(x[2]), reverse=True):
                k = (lbl, mlbl)
                if k not in seen: seen[k] = (lbl, mlbl, u, sig, n)
            return list(seen.values())[:8]

        pos_feats = _dedup(pos_feats)
        neg_feats = _dedup(neg_feats)

        _small_n = len(sub_df) < 20
        if _small_n:
            st.markdown('<div style="font-size:.75rem;color:#906820;margin-bottom:.4rem">⚠ Small selection — showing all computed uplifts including ns (not significant). Treat as directional only.</div>', unsafe_allow_html=True)

        if pos_feats:
            st.markdown('<div style="font-size:.75rem;color:#2A8050;font-weight:600;margin-bottom:.3rem">▲ Positive drivers</div>', unsafe_allow_html=True)
            for _lbl, _mlbl, _u, _sig, _n in pos_feats:
                _sig_col = "#7030A0" if _sig in ("***","**","*") else "#CCC"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:.22rem 0;'
                    f'border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                    f'<span style="color:#1C1C1C"><strong>{_lbl}</strong> on {_mlbl}</span>'
                    f'<span style="display:flex;gap:.4rem;align-items:center">'
                    f'<span style="color:#2A8050;font-weight:600">+{_u:.1f}pp</span>'
                    f'<span style="color:{_sig_col};font-size:.72rem">{_sig}</span>'
                    f'<span style="color:#CCC;font-size:.72rem">n={_n}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-size:.8rem;color:#AAA">No positive drivers found.</p>', unsafe_allow_html=True)

        if neg_feats:
            st.markdown('<div style="font-size:.75rem;color:#C00020;font-weight:600;margin:.7rem 0 .3rem 0">▼ Negative drivers</div>', unsafe_allow_html=True)
            for _lbl, _mlbl, _u, _sig, _n in neg_feats:
                _sig_col = "#7030A0" if _sig in ("***","**","*") else "#CCC"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:.22rem 0;'
                    f'border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                    f'<span style="color:#1C1C1C"><strong>{_lbl}</strong> on {_mlbl}</span>'
                    f'<span style="display:flex;gap:.4rem;align-items:center">'
                    f'<span style="color:#C00020;font-weight:600">{_u:.1f}pp</span>'
                    f'<span style="color:{_sig_col};font-size:.72rem">{_sig}</span>'
                    f'<span style="color:#CCC;font-size:.72rem">n={_n}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True)

    with _fs_col2:
        # Market performance summary
        st.markdown('<div style="font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6A6660;margin-bottom:.6rem">Market performance</div>', unsafe_allow_html=True)
        _ref_mc = "Attention_T2B"
        if _ref_mc in sub_df.columns and "country_name" in sub_df.columns:
            _mkt_means = (sub_df.groupby("country_name")[_ref_mc]
                         .mean().dropna() * 100).sort_values(ascending=False)
            _avg_mean = df_full[_ref_mc].dropna().mean() * 100 if _ref_mc in df_full.columns else None
            _pos_mkts = [(m, v) for m, v in _mkt_means.items() if _avg_mean is None or v >= _avg_mean]
            _neg_mkts = [(m, v) for m, v in _mkt_means.items() if _avg_mean is not None and v < _avg_mean]

            st.markdown(f'<div style="font-size:.72rem;color:#888;margin-bottom:.5rem">Based on Attention vs average ({_avg_mean:.1f}pp)</div>' if _avg_mean else "", unsafe_allow_html=True)

            if _pos_mkts:
                st.markdown('<div style="font-size:.75rem;color:#2A8050;font-weight:600;margin-bottom:.3rem">▲ Above average</div>', unsafe_allow_html=True)
                for _m, _v in _pos_mkts[:8]:
                    _diff = _v - _avg_mean if _avg_mean else 0
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:.22rem 0;' +
                        f'border-bottom:1px solid #F5F3EF;font-size:.8rem">' +
                        f'<span style="color:#1C1C1C">{_m}</span>' +
                        f'<span style="color:#2A8050;font-weight:600">{_v:.1f}pp (+{_diff:.1f}pp)</span></div>',
                        unsafe_allow_html=True)

            if _neg_mkts:
                st.markdown('<div style="font-size:.75rem;color:#C00020;font-weight:600;margin:.7rem 0 .3rem 0">▼ Below average</div>', unsafe_allow_html=True)
                for _m, _v in _neg_mkts[:8]:
                    _diff = _v - _avg_mean if _avg_mean else 0
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:.22rem 0;' +
                        f'border-bottom:1px solid #F5F3EF;font-size:.8rem">' +
                        f'<span style="color:#1C1C1C">{_m}</span>' +
                        f'<span style="color:#C00020;font-weight:600">{_v:.1f}pp ({_diff:.1f}pp)</span></div>',
                        unsafe_allow_html=True)
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
    top_assets = sub_df[sub_df[sort_mc].notna()].sort_values(sort_mc, ascending=False).head(8)

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
          <div class="asset-meta">{r.get("brand_name","")} &nbsp;·&nbsp; {r.get("country_name","")} &nbsp;·&nbsp; {r.get("asset_category","")}</div>
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
    av_options = av_df["asset_name"].tolist()
    if av_options:
        av_sel = st.selectbox("Select asset", av_options, key="asset_viewer_sel")
        av_row = av_df[av_df["asset_name"] == av_sel].iloc[0]

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
            st.markdown('<div style="font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.7rem">Scores for this asset</div>', unsafe_allow_html=True)
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
                    diff_col = GREEN if diff > 0 else (RED if diff < 0 else "#AAA")
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
      <div class="hero-eyebrow">Why · Page 2 of 6</div>
      <div class="hero-title">Feature Impact</div>
      <div class="hero-sub">For each creative feature, the statistical uplift on your
        chosen metric — how much higher or lower the score is when the feature is present.
        Use this page to identify <em>which creative choices drove the results</em> you saw on page 01.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#EAFAF1;border-left:3px solid #2A8050;border-radius:0 4px 4px 0;
      padding:.7rem 1rem;font-size:.82rem;color:#1A4A30;margin-bottom:1.2rem;line-height:1.65;">
      <strong>Analyst workflow:</strong> Select the metric you care most about, then read the table top-to-bottom.
      Focus only on *** and ** findings — these are statistically reliable.
      * is directional; ns should never appear in a brief.
      Open the <strong>Occasions &amp; Passion Point</strong> expander to see which specific values
      (e.g. "Sport / Football" vs "Festive") drive the result — not just whether the feature is present.
      Then go to <strong>03 · Combination Explorer</strong> to test combinations.
    </div>""", unsafe_allow_html=True)

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()


    st.markdown("""<div class="explain-box">
    <strong>How to read this page:</strong> Each bar shows the <em>uplift</em> for that feature —
    the difference in average score between ads that have the feature and ads that don't.
    A bar to the right means the feature is associated with better performance; left means worse.
    Stars show how statistically confident we are: *** = very confident, ** = confident,
    * = directional, ns = not significant (do not brief from ns results).
    The dashed line is always zero — the point where the feature makes no difference.
    </div>""", unsafe_allow_html=True)

    mc_sel=st.selectbox("Metric to display",
                         ["Attention","Persuasion","Likeability",
                          "Experience Recall","Brand Linkage","Uniqueness","Shareability","Tiredness"],
                         key="fi_mc")
    mc_map={"Attention":"Attention_T2B",
            "Persuasion":"Persuasion_T2B","Likeability":"Likeability_Love_Like_T2B",
            "Experience Recall":"Experience_Recall_T2B","Brand Linkage":"Brand_Linkage_T2B",
            "Uniqueness":"Uniqueness_T2B","Shareability":"Shareability_T2B",
            "Tiredness":"Tiredness_T2B"}
    mc=mc_map[mc_sel]

    col_chart, col_alerts = st.columns([1.6,1])

    with col_chart:
        rows=[]
        for feat in ALL_FEATS:
            if feat not in sub_df.columns: continue
            u,sig,n=compute_uplift(sub_df,feat,mc)
            if u is not None:
                rows.append({"feat":feat,"label":FEAT_LABEL.get(feat,feat),"uplift":u,"sig":sig,"n":n})
        if rows:
            fd=pd.DataFrame(rows).sort_values("uplift",ascending=False)
            tbl = """<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.85rem">
            <thead><tr style="border-bottom:2px solid #EAE8E2">
              <th style="text-align:left;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Feature</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Uplift</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Direction</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#7030A0">Significance</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">n (with feature)</th>
            </tr></thead><tbody>"""
            mc_col_active = M_COLOR.get(mc, DARK)
            max_abs = fd["uplift"].abs().max() or 1
            for i, (_, r) in enumerate(fd.iterrows()):
                bg = "#FAFAF8" if i%2==0 else "#FFFFFF"
                u2 = r["uplift"]
                uc = GREEN if u2>=0 else RED
                arrow = "▲" if u2>=0 else "▼"
                sig2 = r["sig"]
                sig_color = "#7030A0" if sig2 in ("***","**","*") else "#AAA"
                # mini bar
                bar_w = int(abs(u2)/max_abs*80)
                bar_dir = "left" if u2>=0 else "right"
                bar_html_str = (f'<div style="display:flex;align-items:center;gap:.3rem;justify-content:{"flex-start" if u2>=0 else "flex-end"}">'
                                f'<div style="width:{bar_w}px;height:6px;background:{uc};border-radius:2px;opacity:.7"></div>'
                                f'</div>')
                tbl += (f'<tr style="background:{bg};border-bottom:1px solid #F0EEE8">'
                        f'<td style="padding:.42rem .6rem;font-weight:600;color:#1C1C1C">{r["label"]}</td>'
                        f'<td style="text-align:center;font-weight:700;color:{uc};font-size:.9rem">{arrow} {abs(u2):.1f}pp</td>'
                        f'<td style="padding:.42rem .2rem">{bar_html_str}</td>'
                        f'<td style="text-align:center;font-weight:600;color:{sig_color}">{sig2}</td>'
                        f'<td style="text-align:center;color:#888">{r["n"]:,}</td>'
                        f'</tr>')
            tbl += "</tbody></table></div>"
            tbl += f'<div style="font-size:.73rem;color:#AAA;margin-top:.4rem">Sorted by {mc_sel} uplift. pp = percentage points. *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, ns = not significant.</div>'
            st.markdown(tbl, unsafe_allow_html=True)
        else:
            st.info("Not enough data to compute feature impact for this scope.")

    with col_alerts:
        st.markdown('<div class="sec-label">Findings summary</div>', unsafe_allow_html=True)

        # Positive feature drivers
        pos_feats = [(FEAT_LABEL.get(r["feat"],r["feat"]), r["uplift"], r["sig"])
                     for r in (rows if rows else [])
                     if r["uplift"] > 0 and r["sig"] in ("***","**","*")]
        pos_feats.sort(key=lambda x: x[1], reverse=True)

        neg_feats = [(FEAT_LABEL.get(r["feat"],r["feat"]), r["uplift"], r["sig"])
                     for r in (rows if rows else [])
                     if r["uplift"] < 0 and r["sig"] in ("***","**","*")]
        neg_feats.sort(key=lambda x: x[1])

        # Market impact
        pos_mkts = []; neg_mkts = []
        if "country_name" in sub_df.columns and sub_df["country_name"].nunique() > 1:
            mkt_scores = (sub_df.groupby("country_name")[mc].mean().dropna() * 100)
            scope_mean_mc = sub_df[mc].dropna().mean() * 100
            for mkt, val in mkt_scores.items():
                diff = val - scope_mean_mc
                if diff > 1.0:   pos_mkts.append((mkt, diff))
                elif diff < -1.0: neg_mkts.append((mkt, diff))
            pos_mkts.sort(key=lambda x: x[1], reverse=True)
            neg_mkts.sort(key=lambda x: x[1])

        def summary_rows(items, color, sign=""):
            if not items:
                return '<div style="font-size:.79rem;color:#AAA;padding:.3rem 0">No significant findings</div>'
            html = ""
            for name, val, *_ in items[:5]:
                html += (f'<div style="display:flex;justify-content:space-between;'
                         f'padding:.28rem 0;border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                         f'<span style="color:#1C1C1C">{name}</span>'
                         f'<span style="font-weight:600;color:{color}">{sign if val>0 else ""}{val:.1f}pp</span>'
                         f'</div>')
            return html

        def mkt_rows(items, color):
            if not items:
                return '<div style="font-size:.79rem;color:#AAA;padding:.3rem 0">No markets above threshold</div>'
            html = ""
            for name, val in items[:5]:
                sign = "+" if val > 0 else ""
                html += (f'<div style="display:flex;justify-content:space-between;'
                         f'padding:.28rem 0;border-bottom:1px solid #F5F3EF;font-size:.8rem">'
                         f'<span style="color:#1C1C1C">{name}</span>'
                         f'<span style="font-weight:600;color:{color}">{sign}{val:.1f}pp</span>'
                         f'</div>')
            return html

        feat_pos_html = summary_rows([(n,v,s) for n,v,s in pos_feats], GREEN, "+")
        feat_neg_html = summary_rows([(n,v,s) for n,v,s in neg_feats], RED)
        mkt_pos_html  = mkt_rows(pos_mkts, GREEN)
        mkt_neg_html  = mkt_rows(neg_mkts, RED)

        st.markdown(f"""
        <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
          color:#2A8050;margin:.5rem 0 .2rem 0">Features driving positive impact</div>
        {feat_pos_html}
        <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
          color:#C00020;margin:.8rem 0 .2rem 0">Features driving negative impact</div>
        {feat_neg_html}
        <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
          color:#2A8050;margin:.8rem 0 .2rem 0">Markets above average</div>
        {mkt_pos_html}
        <div style="font-size:.68rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
          color:#C00020;margin:.8rem 0 .2rem 0">Markets below average</div>
        {mkt_neg_html}
        <div style="font-size:.69rem;color:#AAA;margin-top:.7rem">
          Showing significant features only (*** ** *). Market delta vs scope mean on {mc_sel}. Threshold ±1pp.
        </div>""", unsafe_allow_html=True)

    # ── Per-value breakdown for Occasions and Passion Point (outside columns) ─
    with st.expander("Occasions & Passion Point — per-value impact", expanded=False):
        st.markdown("""<div style="font-size:.82rem;color:#555;margin-bottom:.7rem">
        Each individual Occasion and Passion Point value is tested against assets
        that have <em>no</em> occasion or passion point assigned. This shows which
        specific values drive performance, not just whether the feature is present.</div>""",
        unsafe_allow_html=True)
        for feat_pv, lbl_pv in [("occasions","Occasions"), ("passion_point","Passion Point")]:
            st.markdown(f'<div style="font-size:.75rem;font-weight:700;color:#6A6660;margin:.6rem 0 .35rem 0">{lbl_pv} — uplift on {mc_sel}</div>', unsafe_allow_html=True)
            pv_rows = compute_uplift_per_value(sub_df, feat_pv, mc)
            if pv_rows:
                tbl_pv = '<table style="width:100%;border-collapse:collapse;font-size:.78rem">'
                tbl_pv += '<thead><tr style="border-bottom:1px solid #EAE8E2"><th style="text-align:left;padding:.3rem .4rem;color:#888;font-size:.68rem">Value</th><th style="text-align:center;padding:.3rem .4rem;color:#888;font-size:.68rem">Uplift</th><th style="text-align:center;color:#7030A0;font-size:.68rem">Sig</th><th style="text-align:center;color:#888;font-size:.68rem">n</th></tr></thead><tbody>'
                for idx_pv, rv in enumerate(pv_rows):
                    uc2 = GREEN if rv["uplift"]>=0 else RED
                    bg2 = "#FAFAF8" if idx_pv%2==0 else "#FFF"
                    sc2 = "#7030A0" if rv["sig"] in ("***","**","*") else "#CCC"
                    sign_pv = "+" if rv["uplift"]>=0 else ""
                    tbl_pv += (f'<tr style="background:{bg2};border-bottom:1px solid #F5F3EF">'
                               f'<td style="padding:.28rem .4rem;max-width:300px">{rv["value"][:50]}</td>'
                               f'<td style="text-align:center;font-weight:600;color:{uc2}">{sign_pv}{rv["uplift"]:.1f}pp</td>'
                               f'<td style="text-align:center;color:{sc2}">{rv["sig"]}</td>'
                               f'<td style="text-align:center;color:#AAA">{rv["n"]}</td></tr>')
                tbl_pv += "</tbody></table>"
                st.markdown(tbl_pv, unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#AAA;font-size:.8rem">Not enough data.</p>', unsafe_allow_html=True)

    # Insight catalog entries for this metric
    if has_ins and catalog is not None:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Insight catalog — top findings for this metric</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="explain-box">
        These are statistically validated findings from the full insight pipeline.
        <strong>Confidence</strong> reflects how certain we are: High = safe to brief,
        Medium = directional, Low = do not brief. <strong>Baseline</strong> is the average
        score for assets that do <em>not</em> have this feature — the starting point.</div>""",
        unsafe_allow_html=True)

        smap={"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
        cat_view=catalog[catalog["metric_display"]==mc_sel].copy()
        if scope_filters:
            sl=smap.get(scope_filters[0][0],scope_filters[0][0]); sv=scope_filters[0][1]
            cat_view=cat_view[(cat_view["filter"]==sl)&(cat_view["filter_value"]==sv)]
        else:
            cat_view=cat_view[cat_view["filter"]=="Global"]
        cat_view=cat_view[cat_view["confidence"].isin(["high","medium"])].sort_values(
            "evidence_uplift_pp",ascending=False)

        for _,r in cat_view.head(12).iterrows():
            v=r["evidence_uplift_pp"]; bc=GREEN if v>=0 else RED
            st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
              <div style="display:flex;justify-content:space-between">
                <div class="ic-feat">{r["feature_display"]}</div>
                <div>{bpos(v)}</div>
              </div>
              <div class="ic-text">{r["text"]}</div>
              <div class="ic-meta">
                {conf_badge(r["confidence"])}
                {badge(r["metric_display"],"b-metric")}
                {sig_badge(r["evidence_sig"])}
                <span class="badge b-scope">n={int(r["evidence_n_has"]):,}</span>
                <span class="badge b-ns">base {r["evidence_baseline_pp"]:.1f}pp</span>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 03 — COMBINATION EXPLORER  (dashboard_app logic)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "03 · Combination Explorer":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Why · Page 3 of 6</div>
      <div class="hero-title">Combination Explorer</div>
      <div class="hero-sub">Features rarely work in isolation. This page finds the sequence of
        features that, combined, maximise a metric — and lets you test any custom combination
        in real time across all 8 metrics simultaneously.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#EAFAF1;border-left:3px solid #2A8050;border-radius:0 4px 4px 0;
      padding:.7rem 1rem;font-size:.82rem;color:#1A4A30;margin-bottom:1.2rem;line-height:1.65;">
      <strong>Analyst workflow:</strong> Each metric tab shows up to 3 distinct combinations,
      ordered by total uplift. Combination 1 is the strongest; Combinations 2 and 3 use
      different features so you have alternatives for briefing.
      Use the <strong>Explorer</strong> below to test your own combinations —
      the scoreboard shows all 8 metrics at once so you can spot trade-offs immediately.
      A combination that boosts Attention but increases Tiredness may not be worth briefing.
    </div>""", unsafe_allow_html=True)

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()


    st.markdown("""<div class="explain-box">
    <strong>How the best combination works:</strong> The algorithm starts with the baseline
    (mean score of all assets in scope) and greedily adds one feature at a time — always
    choosing the next feature that adds the most uplift given what's already selected.
    The top <strong>3 features</strong> are shown for each metric. The <strong>Gain</strong>
    column shows each feature's individual contribution. <strong>Total</strong> is the cumulative
    uplift from baseline. <strong>n</strong> is how many assets match all selected features —
    smaller n means the estimate is less certain. <strong>pp</strong> = percentage points.
    </div>""", unsafe_allow_html=True)

    # scope alerts
    alerts=get_scope_alerts(scope_filters)
    if alerts:
        with st.expander(f"⚠ {len(alerts)} scope alert{'s' if len(alerts)>1 else ''}",expanded=False):
            render_alerts(alerts)

    ALL_COMBO_METRICS = [
        ("Attention",        "Attention_T2B"),
        ("Persuasion",       "Persuasion_T2B"),
        ("Likeability",      "Likeability_Love_Like_T2B"),
        ("Exp. Recall",      "Experience_Recall_T2B"),
        ("Brand Linkage",    "Brand_Linkage_T2B"),
        ("Uniqueness",       "Uniqueness_T2B"),
        ("Shareability",     "Shareability_T2B"),
        ("Tiredness",        "Tiredness_T2B"),
    ]
    tab_labels = [m[0] for m in ALL_COMBO_METRICS]
    tabs = st.tabs(tab_labels)
    for (ml, mc), tab in zip(ALL_COMBO_METRICS, tabs):
        with tab:
            # Get top3 from precomputed if available, else fall back to single best
            top3 = []
            if scope_key and scope_key in results and mc in results[scope_key]:
                c = results[scope_key][mc]
                top3 = c.get("top3_combos", [])
                if not top3:
                    # Backward compat: wrap single combo as rank-1
                    if c.get("steps"):
                        top3 = [{"rank":1,"combo":c["combo"],"steps":c["steps"],
                                  "baseline_pp":c["baseline_pp"],"n_total":c["n_total"]}]

            if not top3:
                st.markdown('<p style="color:#CCC8C2;padding:1.5rem 0;font-size:.84rem">No combination found for this scope.</p>',
                            unsafe_allow_html=True)
                continue

            baseline_pp = top3[0]["baseline_pp"]
            n_total     = top3[0]["n_total"]

            for rank_data in top3:
                steps     = rank_data["steps"]
                combo     = rank_data["combo"]
                rank      = rank_data["rank"]
                if not steps:
                    continue
                final_mean   = steps[-1]["metric_mean"]
                total_uplift = steps[-1]["cumulative_uplift_pp"]
                final_n      = steps[-1]["n"]

                # ── Combination {rank} ───────────────────────────────────────────
                st.markdown(f'<div style="font-size:.78rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#6A6660;margin:1.2rem 0 .4rem 0">Combination {rank} of 3</div>', unsafe_allow_html=True)
                col_wf,col_right=st.columns([1.5,1])
                with col_wf:
                    # combo summary card
                    st.markdown(f"""
                    <div class="combo-card">
                      <div style="display:flex;align-items:flex-end;gap:.9rem;margin-bottom:.25rem">
                        <span class="combo-uplift">+{total_uplift:.1f}</span>
                        <span style="font-family:'Merriweather',serif;font-size:1.3rem;color:#D0C8C0;font-weight:300">pp</span>
                        <div style="padding-bottom:.22rem">
                          <div style="font-size:.74rem;color:#AAA89E">cumulative uplift on {ml}</div>
                          <div style="font-size:.68rem;color:#CCC8C2;margin-top:.08rem">
                            Baseline {baseline_pp:.1f}pp &rarr; {final_mean:.1f}pp &nbsp;&middot;&nbsp;
                            {len(steps)} features &nbsp;&middot;&nbsp; n={final_n:,}
                          </div>
                        </div>
                      </div>""", unsafe_allow_html=True)
                    pills='<div class="pill-row">'
                    for s in steps:
                        pills+=(f'<div class="pill"><span class="pill-n">{s["step"]}</span>'
                                f'{s["label"]}<span class="pill-gain">+{s["step_gain_pp"]:.1f}pp</span></div>')
                    st.markdown(pills+"</div></div>",unsafe_allow_html=True)
                    if final_n<10:
                        st.markdown(f'<div class="low-n">Only {final_n} assets match — treat as directional.</div>',
                                    unsafe_allow_html=True)
    
                    # waterfall
                    st.markdown('<div style="margin-top:1.4rem"></div>',unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="wf-row" style="border-bottom:1px solid #EAE8E2;padding-bottom:.2rem">
                      <div class="wf-idx"></div><div class="wf-label wf-col">Feature</div>
                      <div class="wf-bar wf-col">Gain vs previous step</div>
                      <div class="wf-n wf-col">n</div><div class="wf-pp wf-col">Total</div>
                    </div>
                    <div class="wf-row">
                      <div class="wf-idx">—</div>
                      <div class="wf-label" style="color:#CCC8C2;font-style:italic">Baseline — {n_total:,} assets in scope ({baseline_pp:.1f}pp)</div>
                      <div class="wf-bar">{bar_html(0)}</div>
                      <div class="wf-n">{n_total:,}</div>
                      <div class="wf-pp" style="color:#CCC8C2">{baseline_pp:.1f}</div>
                    </div>""",unsafe_allow_html=True)
                    mx=max(s["step_gain_pp"] for s in steps) if steps else 1
                    for s in steps:
                        frac=s["step_gain_pp"]/mx if mx>0 else 0
                        st.markdown(f"""
                        <div class="wf-row">
                          <div class="wf-idx" style="color:#E8002D;font-weight:600">{s["step"]}</div>
                          <div class="wf-label">+ {s["label"]}</div>
                          <div class="wf-bar">{bar_html(frac)}</div>
                          <div class="wf-n">{s["n"]:,}</div>
                          <div class="wf-pp">+{s["cumulative_uplift_pp"]:.1f}</div>
                        </div>""",unsafe_allow_html=True)
    
                with col_right:
                    # alerts for top combo feature
                    if steps:
                        top_f=steps[0]["feat"]
                        st.markdown(f'<div class="sec-label">Alerts — {FEAT_LABEL.get(top_f,top_f)}</div>',
                                    unsafe_allow_html=True)
                        fit=get_feature_alerts(top_f,scope_filters,mc)
                        if fit: render_alerts(fit,max_items=5)
                        else: st.markdown('<p style="color:#AAA;font-size:.81rem">No alerts for this feature.</p>',unsafe_allow_html=True)
    
                    # Top assets matching combination
                    st.markdown('<div style="margin-top:1rem"></div>',unsafe_allow_html=True)
                    st.markdown('<div class="sec-label">Top assets matching this combination</div>',unsafe_allow_html=True)
                    if combo:
                        sel_tmp=default_sel(combo)
                        mask_tmp=apply_sel(sub_df,sel_tmp)
                        top_tmp=sub_df.loc[mask_tmp].sort_values("Attention_T2B",ascending=False).head(4)
                        for _,r in top_tmp.iterrows():
                            url=r.get("asset_url",""); nm=r.get("asset_name",f"Asset {r.get('asset_sk_id','')}")
                            lnk=f'<a href="{url}" target="_blank" style="color:{RED};font-size:.71rem;font-weight:600;text-decoration:none">▶ View</a>' if url else ""
                            att=f"{r['Attention_T2B']*100:.0f}pp" if pd.notna(r.get("Attention_T2B")) else "—"
                            st.markdown(f"""<div class="asset-card">
                              <div style="display:flex;justify-content:space-between">
                                <div class="asset-name" style="font-size:.76rem">{nm[:50]}</div>{lnk}
                              </div>
                              <div class="asset-meta">Att {att} &nbsp;·&nbsp; {r.get("country_name","")}</div>
                            </div>""",unsafe_allow_html=True)
    
            # ── Feature Explorer ─────────────────────────────────────────────
            st.markdown("""<div class="explorer-card">
              <div class="explorer-title">Explore any combination</div>
              <div class="explorer-sub">
                Pre-filled with the best combination above. Change any feature to see all
                four metrics update in real time. The scoreboard shows your selection's
                average score vs the scope mean. Feature values include their pp uplift
                so you can see which specific values drive performance.
              </div>""",unsafe_allow_html=True)

            ss=f"sel_{mc}_{hash(str(scope_filters))}_{sel_camp_i}"
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
                        try: ci2=raws.index(cr)
                        except ValueError: ci2=0
                        ch=st.selectbox(FEAT_LABEL.get(feat,feat),range(len(raws)),
                                         format_func=lambda i,d=disps:d[i],index=ci2,
                                         key=f"d_{mc}_{feat}_{hash(str(scope_filters))}_{sel_camp_i}")
                        chv=raws[ch]; new[feat]=chv
                        if chv!=cr: changed=True
                for ci in range(len(row_feats),5): cols[ci].empty()

            if changed: st.session_state[ss]=new; st.rerun()

            r1,r2,_=st.columns([1,1,7])
            with r1:
                if st.button("↺ Reset",key=f"rst_{mc}_{hash(str(scope_filters))}_{sel_camp_i}"):
                    st.session_state[ss]=default_sel(combo) if combo else {f:"__any__" for f in ALL_FEATS}
                    st.rerun()
            with r2:
                if st.button("✕ Clear",key=f"clr_{mc}_{hash(str(scope_filters))}_{sel_camp_i}"):
                    st.session_state[ss]={f:"__any__" for f in ALL_FEATS}; st.rerun()

            # Scoreboard
            scores=score_sel(sub_df,cur)
            bases={mc2:round(sub_df[mc2].dropna().mean()*100,2) for mc2 in ALL_METRICS if mc2 in sub_df.columns}
            sb='<div class="scoreboard">'
            for mc2 in ALL_METRICS:
                if mc2 not in sub_df.columns: continue
                ml2 = M_LABEL.get(mc2, METRICS.get(mc2, mc2))
                val,nv=scores.get(mc2,(None,0)); base=bases.get(mc2,0)
                hi=" hi" if mc2==mc else ""; gc=M_COLOR.get(mc2,DARK)
                if val is not None:
                    d=val-base; dc="d-up" if d>.1 else("d-dn" if d<-.1 else"d-flat")
                    sb+=(f'<div class="score-cell{hi}" style="border-top:2px solid {gc}">'
                         f'<div class="score-metric">{ml2}</div>'
                         f'<div class="score-val">{val:.1f}</div>'
                         f'<div class="score-delta {dc}">{"+" if d>0 else ""}{d:.1f}pp vs mean</div>'
                         f'<div class="score-n">n={nv:,}</div></div>')
                else:
                    sb+=(f'<div class="score-cell{hi}" style="border-top:2px solid {gc}">'
                         f'<div class="score-metric">{ml2}</div>'
                         f'<div class="score-val" style="font-size:.9rem;color:#D8D4CE">—</div>'
                         f'<div class="score-n">n=0</div></div>')
            st.markdown(sb+"</div>",unsafe_allow_html=True)

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
                                '<div class="sec-label">Insights for your selection</div>',unsafe_allow_html=True)
                    render_alerts(items[:5]); st.markdown("</div>",unsafe_allow_html=True)

            st.markdown("</div>",unsafe_allow_html=True)  # close explorer-card


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 04 — FEATURE COMBINATIONS & OU IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "04 · Feature Combinations & OU Impact":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Why · Page 4 of 6</div>
      <div class="hero-title">Feature Combinations &amp; OU Impact</div>
      <div class="hero-sub">Two questions in one page: which features amplify each other's effect
        (synergy analysis), and does the same feature perform consistently across Operating Units?
        A positive feature on average can reverse in specific markets.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#EAFAF1;border-left:3px solid #2A8050;border-radius:0 4px 4px 0;
      padding:.7rem 1rem;font-size:.82rem;color:#1A4A30;margin-bottom:1.2rem;line-height:1.65;">
      <strong>Analyst workflow:</strong> Select a feature you want to brief, then check
      <strong>Part A</strong> for which other features amplify it (positive synergy = brief both together).
      Then check <strong>Part B</strong> — if any OU bar is red, that OU needs a different brief.
      Never brief a feature without checking Part B first.
      The best-combination-per-OU table shows what to recommend locally when the dataset average doesn't apply.
    </div>""", unsafe_allow_html=True)

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()


    feat_opts=[(f,FEAT_LABEL.get(f,f)) for f in BIN_FEATS if f in sub_df.columns]
    feat_labels=[fl for _,fl in feat_opts]
    feat_cols=[fc for fc,_ in feat_opts]
    fi=st.selectbox("Select a feature to analyse",range(len(feat_labels)),
                     format_func=lambda i:feat_labels[i],key="combo_feat")
    sel_f=feat_cols[fi]; sel_fl=feat_labels[fi]

    mc_c=st.selectbox("Metric",["Attention","Persuasion","Likeability",
                                 "Experience Recall","Brand Linkage","Uniqueness","Shareability","Tiredness"],key="combo_mc")
    mc_c_col={"Attention":"Attention_T2B",
               "Persuasion":"Persuasion_T2B","Likeability":"Likeability_Love_Like_T2B",
               "Experience Recall":"Experience_Recall_T2B","Brand Linkage":"Brand_Linkage_T2B",
               "Uniqueness":"Uniqueness_T2B","Shareability":"Shareability_T2B",
               "Tiredness":"Tiredness_T2B"}[mc_c]

    # Solo uplift
    u_s,sig_s,n_s=compute_uplift(sub_df,sel_f,mc_c_col)

    col_combo,col_ou=st.columns([1.2,1])

    with col_combo:
        # ── Section A: best combinations ─────────────────────────────────────
        st.markdown(f'<div class="sec-label">Part A — Best combinations with {sel_fl}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="explain-box">
        <strong>Solo uplift of {sel_fl}:</strong>
        {f"{u_s:+.1f}pp on {mc_c} ({sig_s}, n={n_s:,})" if u_s is not None else "Insufficient data"}<br><br>
        The table below shows which other binary features, when paired with <strong>{sel_fl}</strong>,
        produce the highest combined score. <strong>Combined uplift</strong> is the expected gain
        when an asset has <em>both</em> features vs the scope average.
        <strong>Synergy</strong> is the extra gain from pairing beyond the best individual feature alone —
        a positive synergy means the two features amplify each other.</div>""",
        unsafe_allow_html=True)

        combos=feature_combinations(sub_df,sel_f,mc_c_col,top_n=7)
        if combos:
            tbl="""<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.83rem">
            <thead><tr style="border-bottom:2px solid #EAE8E2">
              <th style="text-align:left;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Partner feature</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Combined uplift</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Partner alone</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#7030A0">Synergy</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">n assets</th>
              <th style="text-align:left;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Reading</th>
            </tr></thead><tbody>"""
            for ci,c in enumerate(combos):
                bg="#FAFAF8" if ci%2==0 else "#FFF"
                cu=c["combined"]; sy=c["synergy"]
                cuc=GREEN if cu>=0 else RED; syc=GREEN if sy>=0 else RED
                interp=("Strong synergy — better together" if sy>=1.5
                        else ("Additive — modest extra gain" if sy>=0
                              else "Diminishing — one may suffice"))
                ic="#1A7040" if sy>=1.5 else ("#555" if sy>=0 else "#906820")
                tbl+=(f'<tr style="background:{bg};border-bottom:1px solid #F0EEE8">'
                      f'<td style="padding:.42rem .6rem;font-weight:600">{c["label"]}</td>'
                      f'<td style="text-align:center;font-weight:600;color:{cuc}">{"+" if cu>=0 else ""}{cu:.1f}pp</td>'
                      f'<td style="text-align:center;color:#555">{"+" if c["solo_p"]>=0 else ""}{c["solo_p"]:.1f}pp</td>'
                      f'<td style="text-align:center;font-weight:600;color:{syc}">{"+" if sy>=0 else ""}{sy:.1f}pp</td>'
                      f'<td style="text-align:center;color:#AAA">{c["n"]}</td>'
                      f'<td style="font-size:.77rem;font-style:italic;color:{ic}">{interp}</td></tr>')
            tbl+="</tbody></table></div>"
            tbl+='<div style="font-size:.73rem;color:#AAA;margin-top:.4rem">Combined uplift = mean score when both features present, vs scope mean. Synergy = combined uplift minus the better of the two individual uplifts.</div>'
            st.markdown(tbl,unsafe_allow_html=True)
        else:
            st.info("Not enough data to compute combinations in this scope.")

        # Rulebook alerts for this feature
        st.markdown('<div style="margin-top:1.1rem"></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">Rulebook alerts — {sel_fl}</div>',unsafe_allow_html=True)
        fit=get_feature_alerts(sel_f,scope_filters,mc_c_col,max_items=5)
        if fit: render_alerts(fit,max_items=5)
        else: st.markdown('<p style="color:#AAA;font-size:.81rem">No alerts for this feature.</p>',unsafe_allow_html=True)

    with col_ou:
        # ── Section B: OU impact ──────────────────────────────────────────────
        st.markdown(f'<div class="sec-label">Part B — OU impact for {sel_fl}</div>',unsafe_allow_html=True)
        st.markdown(f"""<div class="explain-box">
        The same feature can perform very differently across Operating Units.
        Each bar shows the {mc_c} uplift for <strong>{sel_fl}</strong> within that OU.
        The blue dotted line is the dataset average. A red bar means the feature is
        associated with <em>lower</em> scores in that OU — critical to check before
        briefing a positive feature on average.</div>""",unsafe_allow_html=True)

        ou_rows=[]
        for ou in sorted(df_full["operating_unit_code"].dropna().unique()):
            ou_sub=df_full[df_full["operating_unit_code"]==ou]
            if scope_filters:
                for t,v in scope_filters:
                    if t!="ou": ou_sub=ou_sub[ou_sub[SCOPE_COL[t]]==v]
            u2,sig2,n2=compute_uplift(ou_sub,sel_f,mc_c_col)
            if u2 is not None: ou_rows.append({"OU":ou,"uplift":u2,"sig":sig2,"n":n2})
        if ou_rows:
            ou_df=pd.DataFrame(ou_rows).sort_values("uplift")
            g_u,_,_=compute_uplift(df_full,sel_f,mc_c_col)
            fig,ax=plt.subplots(figsize=(6,max(3.5,len(ou_df)*.5)))
            fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
            ax.barh(ou_df["OU"],ou_df["uplift"],
                     color=[GREEN if v>=0 else RED for v in ou_df["uplift"]],
                     alpha=.85,height=.62)
            ax.axvline(0,color=DARK,lw=1.2,ls="--")
            if g_u is not None:
                ax.axvline(g_u,color=BLUE,lw=1.5,ls=":",alpha=.75,label=f"Avg: {g_u:+.1f}pp")
                ax.legend(fontsize=8)
            for i,(_,r) in enumerate(ou_df.iterrows()):
                v=r["uplift"]
                ax.text(v+(.12 if v>=0 else -.12),i,f"{v:+.1f} {r['sig']}",
                        va="center",ha="left" if v>=0 else "right",fontsize=8,color=DARK)
            ax.set_xlabel(f"{sel_fl} uplift on {mc_c} (pp)",fontsize=8.5)
            ax.set_title(f"{sel_fl} — {mc_c} by OU",fontsize=9,fontweight="bold",color=DARK)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#DDD")
            ax.grid(axis="x",alpha=.3); plt.tight_layout()
            st.pyplot(fig); plt.close()

            # best combos per OU
            st.markdown('<div style="margin-top:1rem"></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="sec-label">Best combination per OU — {sel_fl} + top partner</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="explain-box" style="font-size:.78rem">
            For each OU, the best partner feature to combine with the selected feature on Attention.</div>""",
            unsafe_allow_html=True)
            ou_combo_rows=[]
            for ou in sorted(df_full["operating_unit_code"].dropna().unique()):
                ou_sub2=df_full[df_full["operating_unit_code"]==ou]
                if scope_filters:
                    for t,v in scope_filters:
                        if t!="ou": ou_sub2=ou_sub2[ou_sub2[SCOPE_COL[t]]==v]
                cc=feature_combinations(ou_sub2,sel_f,"Attention_T2B",top_n=1)
                if cc:
                    ou_combo_rows.append({"OU":ou,"Partner":cc[0]["label"],
                                          "Combined Uplift":f'+{cc[0]["combined"]:.1f}pp',
                                          "Synergy":f'+{cc[0]["synergy"]:.1f}pp' if cc[0]["synergy"]>=0
                                                    else f'{cc[0]["synergy"]:.1f}pp',
                                          "n":cc[0]["n"]})
            if ou_combo_rows:
                st.dataframe(pd.DataFrame(ou_combo_rows).set_index("OU"),)
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
      <div class="hero-eyebrow">So What · Page 5 of 6</div>
      <div class="hero-title">Insight Catalog</div>
      <div class="hero-sub">The evidence base for creative briefs. Every statistically tested
        finding — filterable by scope, metric, and confidence level.
        High-confidence findings are safe to brief directly. Always check the market impact
        section before recommending a finding for a specific region.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#F3EEFF;border-left:3px solid #7030A0;border-radius:0 4px 4px 0;
      padding:.7rem 1rem;font-size:.82rem;color:#3A1A6A;margin-bottom:1.2rem;line-height:1.65;">
      <strong>Analyst workflow:</strong> Filter by <strong>Confidence: High</strong> to get findings
      safe for briefing. Use the Scope dropdown (OU or Brand) to check if findings are specific
      to your context or truly global. Open <strong>Market impact</strong> to see whether a
      finding reverses in any market. The feature panel above the results shows asset prevalence
      in scope — useful for understanding how common each feature is in practice.
    </div>""", unsafe_allow_html=True)

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()

    with st.expander("How to use this catalog"):
        st.markdown("""
**Confidence** tells you how much to trust a finding:
- **High** — significant at ** or ***, at least 50 assets on both sides, no sample size warning. Safe to brief.
- **Medium** — significant at any level, at least 20 assets. Use as directional guidance.
- **Low** — not significant, or very small sample. Do not brief from this.

**Uplift (pp)** — how much higher (▲) or lower (▼) the average score is for assets with this feature vs assets without it, in percentage points.

**Baseline** — the average score of assets that do *not* have this feature. This is the starting point.

**Significance** — *** means less than 1-in-1,000 chance the result is noise. ** = 1-in-100. * = 1-in-20. ns = not significant.

**Quality flags** — small_sample (fewer than 30 assets on one side), multiple_testing_risk (many features tested at once), low_value_presence (very few assets have this feature).
        """)

    # Filters — Scope limited to Brand and OU only
    cf1,cf2,cf3,cf4,cf5 = st.columns(5)
    with cf1:
        default_scope="All"
        if scope_filters:
            smap2={"ou":"OU","brand":"Brand"}
            sl=smap2.get(scope_filters[0][0],""); sv=scope_filters[0][1]
            if sl: default_scope=f"{sl}: {sv}"
        scope_opts=["All"]+sorted(
            [f"{r['filter']}: {r['filter_value']}" for _,r in
             catalog[["filter","filter_value"]].drop_duplicates().iterrows()
             if r["filter"] in ("OU","Brand","Market")])
        cat_scope=st.selectbox("Scope",scope_opts,
                                index=scope_opts.index(default_scope) if default_scope in scope_opts else 0)
    with cf2:
        cat_metric=st.multiselect("Metric",sorted(catalog["metric_display"].unique()),
                                   default=sorted(catalog["metric_display"].unique()))
    with cf3:
        cat_conf=st.multiselect("Confidence",["high","medium","low"],default=["high","medium"])
    with cf4:
        cat_dir=st.radio("Direction",["Both","Positive","Negative"])
    with cf5:
        cat_sort=st.selectbox("Sort",["Highest uplift","Lowest uplift","Confidence","Feature"])

    fc=catalog.copy()
    if cat_scope!="All":
        parts=cat_scope.split(": ",1)
        if len(parts)==2: fc=fc[(fc["filter"]==parts[0])&(fc["filter_value"]==parts[1])]
    if cat_metric: fc=fc[fc["metric_display"].isin(cat_metric)]
    if cat_conf:   fc=fc[fc["confidence"].isin(cat_conf)]
    if cat_dir=="Positive": fc=fc[fc["direction"]=="positive"]
    elif cat_dir=="Negative": fc=fc[fc["direction"]=="negative"]
    if cat_sort=="Highest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=False)
    elif cat_sort=="Lowest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=True)
    elif cat_sort=="Confidence":
        fc["_co"]=fc["confidence"].map({"high":0,"medium":1,"low":2})
        fc=fc.sort_values(["_co","evidence_uplift_pp"],ascending=[True,False])
    elif cat_sort=="Feature": fc=fc.sort_values("feature_display")

    st.markdown(f"<span style='color:{MID};font-size:.83rem'>{len(fc):,} insights</span>",unsafe_allow_html=True)

    # ── Asset Feature Panel ───────────────────────────────────────────────────
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Asset features &amp; their values in this scope</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    All feature columns and the values they take across assets in the current scope.
    Binary features show how many assets have the feature. Categorical features list every value and asset count.</div>""",
    unsafe_allow_html=True)

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

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── Market impact comparison ──────────────────────────────────────────────
    with st.expander("Market impact — how do findings differ across markets?", expanded=False):
        st.markdown("""<div style="font-size:.82rem;color:#555;margin-bottom:.7rem">
        Select a feature and metric to see how the uplift varies across markets.
        Red bars = the feature hurts performance in that market (reversal).
        Always check for reversals before briefing a positive feature on average locally.</div>""",
        unsafe_allow_html=True)
        mkt_feat_opts = [(f, FEAT_LABEL.get(f,f)) for f in BIN_FEATS if f in sub_df.columns]
        mkt_fi = st.selectbox("Feature", range(len(mkt_feat_opts)),
                               format_func=lambda i: mkt_feat_opts[i][1], key="mkt_ins_feat")
        mkt_mc_sel = st.selectbox("Metric", list({v:k for k,v in M_LABEL.items()}.keys()), key="mkt_ins_mc")
        mkt_mc_col2 = {v:k for k,v in M_LABEL.items()}.get(mkt_mc_sel, "Attention_T2B")
        mkt_feat_col, _ = mkt_feat_opts[mkt_fi]
        mkt_rows_ins = []
        for mkt in sorted(df_full["country_name"].dropna().unique()):
            mkt_sub2 = df_full[df_full["country_name"]==mkt]
            u2,sig2,n2 = compute_uplift(mkt_sub2, mkt_feat_col, mkt_mc_col2)
            if u2 is not None: mkt_rows_ins.append({"Market":mkt,"Uplift (pp)":round(u2,1),"Sig":sig2,"n":n2})
        if mkt_rows_ins:
            mkt_ins_df = pd.DataFrame(mkt_rows_ins).sort_values("Uplift (pp)", ascending=False)
            g_u2,_,_ = compute_uplift(df_full, mkt_feat_col, mkt_mc_col2)
            tbl_mkt_ins = '<table style="width:100%;border-collapse:collapse;font-size:.8rem"><thead><tr style="border-bottom:2px solid #EAE8E2"><th style="text-align:left;padding:.35rem .5rem;color:#888">Market</th><th style="text-align:center;color:#888">Uplift</th><th style="text-align:center;color:#7030A0">Sig</th><th style="text-align:center;color:#888">n</th><th style="text-align:left;color:#888">Signal</th></tr></thead><tbody>'
            for i,(_, rm) in enumerate(mkt_ins_df.iterrows()):
                bg3 = "#FAFAF8" if i%2==0 else "#FFF"
                uc3 = GREEN if rm["Uplift (pp)"]>=0 else RED
                sig3 = rm["Sig"]
                signal = ""
                if g_u2 is not None and rm["Uplift (pp)"] < 0 < g_u2 and sig3 in ("*","**","***"):
                    signal = '<span style="color:#C00020;font-size:.72rem">⚠ Reversal</span>'
                elif sig3 in ("**","***") and rm["Uplift (pp)"] > 0:
                    signal = '<span style="color:#2A8050;font-size:.72rem">✓ Positive</span>'
                tbl_mkt_ins += (f'<tr style="background:{bg3};border-bottom:1px solid #F5F3EF">'
                                f'<td style="padding:.28rem .5rem">{rm["Market"]}</td>'
                                f'<td style="text-align:center;font-weight:600;color:{uc3}">{"+" if rm["Uplift (pp)"]>=0 else ""}{rm["Uplift (pp)"]:.1f}pp</td>'
                                f'<td style="text-align:center;color:#7030A0">{sig3}</td>'
                                f'<td style="text-align:center;color:#AAA">{rm["n"]}</td>'
                                f'<td>{signal}</td></tr>')
            tbl_mkt_ins += '</tbody></table>'
            if g_u2 is not None:
                st.markdown(f'<div style="font-size:.78rem;color:#555;margin-bottom:.4rem">Average: <strong>{"+" if g_u2>=0 else ""}{g_u2:.1f}pp</strong></div>', unsafe_allow_html=True)
            st.markdown(tbl_mkt_ins, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#AAA;font-size:.8rem">Not enough market data.</p>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── View toggle — Table default ───────────────────────────────────────────
    vmode=st.radio("View",["Table","Cards"])

    if vmode=="Table":
        sc2=["feature_display","metric_display","filter","filter_value",
             "evidence_uplift_pp","evidence_sig","confidence","effect_size",
             "evidence_n_has","evidence_baseline_pp","quality_flags"]
        st.dataframe(fc[[c for c in sc2 if c in fc.columns]].rename(columns={
            "feature_display":"Feature","metric_display":"Metric","filter":"Scope",
            "filter_value":"Scope value","evidence_uplift_pp":"Uplift (pp)",
            "evidence_sig":"Sig.","confidence":"Confidence","effect_size":"Effect size",
            "evidence_n_has":"n (has)","evidence_baseline_pp":"Baseline (pp)",
            "quality_flags":"Flags"}).reset_index(drop=True),height=600)
    else:
        pg=max(1,int(np.ceil(len(fc)/20)))
        pnum=st.slider("Page",1,pg,1) if pg>1 else 1
        for _,r in fc.iloc[(pnum-1)*20:pnum*20].iterrows():
            v=r["evidence_uplift_pp"]; bc=GREEN if v>=0 else RED
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
            cb=feature_combinations(sub_df,r["feature"],
                                     {"Attention":"Attention_T2B","Persuasion":"Persuasion_T2B",
                                      "Likeability":"Likeability_Love_Like_T2B",
                                      "Experience Recall":"Experience_Recall_T2B",
                                      "Brand Linkage":"Brand_Linkage_T2B",
                                      "Uniqueness":"Uniqueness_T2B","Shareability":"Shareability_T2B",
                                      "Tiredness":"Tiredness_T2B"}.get(r["metric_display"],"Attention_T2B"),
                                     top_n=1)
            combo_snippet=""
            if cb:
                combo_snippet=(f'<div style="margin-top:5px;font-size:.76rem;color:#555">'
                               f'Best combination: <strong>{cb[0]["label"]}</strong> → '
                               f'{cb[0]["combined"]:+.1f}pp combined (synergy {cb[0]["synergy"]:+.1f}pp)</div>')
            st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
              <div style="display:flex;justify-content:space-between">
                <div class="ic-feat">{r["feature_display"]}</div>
                <div>{bpos(v)}</div>
              </div>
              <div class="ic-text">{r["text"]}</div>
              {ou_snippet}{combo_snippet}
              <div class="ic-meta">
                {conf_badge(r["confidence"])}
                {badge(r["metric_display"],"b-metric")}
                {sig_badge(r["evidence_sig"])}
                <span class="badge b-scope">n={int(r["evidence_n_has"]):,}</span>
                <span class="badge b-ns">base {r["evidence_baseline_pp"]:.1f}pp</span>
                {badge(r.get("filter","")+": "+r.get("filter_value",""),"b-scope")}
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
      <div class="hero-eyebrow">So What · Page 6 of 6</div>
      <div class="hero-title">Rulebook</div>
      <div class="hero-sub">Automated risk and opportunity detection across all scopes.
        Seven alert types surface trade-offs, market reversals, underused features, and
        anti-patterns before they end up in a brief.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#F3EEFF;border-left:3px solid #7030A0;border-radius:0 4px 4px 0;
      padding:.7rem 1rem;font-size:.82rem;color:#3A1A6A;margin-bottom:1.2rem;line-height:1.65;">
      <strong>Analyst workflow:</strong> Filter by <strong>Severity: High</strong> first —
      these are the alerts that can invalidate a brief if missed.
      <strong>Heterogeneity</strong> alerts are the most critical: they mean a globally
      positive feature reverses in a specific OU or market.
      <strong>Conflict</strong> alerts flag metric trade-offs — brief with the dominant metric
      for this campaign objective in mind.
      <strong>Opportunity</strong> alerts point to features worth testing more.
    </div>""", unsafe_allow_html=True)

    sub_df, scope_filters, scope_key, min_n, scope_label, sel_camp_i = render_filters()
    if len(sub_df) < 3:
        st.warning(f"Only {len(sub_df)} assets match this filter — too few for analysis.")
        st.stop()


    with st.expander("What does each rule type mean?"):
        st.markdown("""
| Rule | What it means | What to do |
|---|---|---|
| **Conflict** | Feature improves one metric but hurts another | Check the trade-off — choose which metric matters more for this campaign |
| **Heterogeneity** | Feature is positive globally but negative in a specific scope | Never brief from the dataset average alone — always check the specific OU/market first |
| **Boundary Condition** | Feature works in Video but not Print, or vice versa | Brief this feature only for the format where it works |
| **Opportunity** | High positive effect but the feature appears in fewer than 20% of assets | This feature works — test more assets with it |
| **Outlier** | One scope is unusually different from its peers | Investigate before briefing — the difference may be cultural or a tagging anomaly |
| **Consensus** | Feature is positive across all three metrics | Safest feature to brief — no trade-offs |
| **Anti-pattern** | Feature is negative across all three metrics in a scope | Brief against this feature in this specific scope |
        """)

    rule_counts=rulebook["rule_type"].value_counts()
    RULE_BG={"Conflict":"#FFF0F0","Heterogeneity":"#FFFBF0","Boundary Condition":"#F8F0FF",
              "Opportunity":"#F0FFF4","Outlier":"#EBF3FB","Consensus":"#F0FFF4","Anti-pattern":"#FFF0F0"}
    RULE_AC={"Conflict":RED,"Heterogeneity":"#906820","Boundary Condition":"#7030A0",
              "Opportunity":"#1A7040","Outlier":"#1A3080","Consensus":"#1A7040","Anti-pattern":RED}
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

    rf1,rf2,rf3=st.columns(3)
    with rf1: rt_f=st.multiselect("Rule type",sorted(rulebook["rule_type"].unique()),
                                   default=sorted(rulebook["rule_type"].unique()))
    with rf2: sv_f=st.multiselect("Severity",["high","medium","low"],default=["high","medium"])
    with rf3: sc_f=st.multiselect("Scope",sorted(rulebook["scope"].unique()),
                                   default=sorted(rulebook["scope"].unique()))

    rb_f=rulebook.copy()
    if rt_f: rb_f=rb_f[rb_f["rule_type"].isin(rt_f)]
    if sv_f: rb_f=rb_f[rb_f["severity"].isin(sv_f)]
    if sc_f: rb_f=rb_f[rb_f["scope"].isin(sc_f)]
    st.markdown(f"<span style='color:{MID};font-size:.83rem'>{len(rb_f)} entries</span>",unsafe_allow_html=True)

    SEV_COL={"high":RED,"medium":"#906820","low":"#2A8050"}
    for _,r in rb_f.iterrows():
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
          <div style="display:flex;gap:5px;margin-top:.4rem;flex-wrap:wrap">
            {badge(r.get("feature",""),"b-metric")}
            {badge(r.get("scope",""),"b-scope")}
            {badge(r.get("scope_value","") or "Global","b-ns")}
          </div>
        </div>""",unsafe_allow_html=True)


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>Uplift = mean(feature present) &minus; mean(feature absent) &nbsp;&middot;&nbsp;
    Significance: Mann-Whitney U &nbsp;&middot;&nbsp; pp = percentage points &nbsp;&middot;&nbsp;
    SCD = 10% See + 30% Connect + 60% Do</span>
  <span>The Coca&#8209;Cola Company &nbsp;&middot;&nbsp; Asset Intelligence</span>
</div>
""", unsafe_allow_html=True)
