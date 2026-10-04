"""
OR Assistant - Main Streamlit Application
AI-Powered Operations Research Tool
"""

import streamlit as st
from dotenv import load_dotenv
from copy import deepcopy
import os
import re as _re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Import project modules (these will be implemented)
try:
    from src.agents.problem_classifier import ProblemClassifier
    from src.modeling.model_generator import ModelGenerator
    from src.solvers.solver_interface import SolverInterface
    from src.interpreters.result_interpreter import ResultInterpreter
    from src.ingestion.file_parser import FileParser
    from src.ingestion.data_extractor import DataExtractor
    from src.ingestion.miplib_loader import MIPLIBLoader, RECOMMENDED_INSTANCES
except ImportError:
    st.warning("⚠️ Some modules are not yet implemented. See development guide.")

# Page configuration
st.set_page_config(
    page_title="OR Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
    }

    /* ================================================================
       MODEL PLAYGROUND — Scoped via :has(.or-playground)
       Frosted-glass cards, blue accent, clear elevation hierarchy
       ================================================================ */

    /* ── Entrance ── */
    @keyframes orFadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    [role="tabpanel"]:has(.or-playground) > div {
      animation: orFadeIn 250ms ease-out both;
    }

    /* ── Background gradient on the tab panel ── */
    [role="tabpanel"]:has(.or-playground) {
      background: linear-gradient(160deg, #161b2e 0%, #0f1219 60%, #141824 100%);
      border-radius: 12px;
      padding: 8px;
    }

    /* ── Typography ── */
    [role="tabpanel"]:has(.or-playground) h1,
    [role="tabpanel"]:has(.or-playground) h2,
    [role="tabpanel"]:has(.or-playground) h3 {
      font-weight: 600;
      letter-spacing: -0.02em;
      color: #e8ecf4;
    }
    [role="tabpanel"]:has(.or-playground) h4 {
      font-weight: 600;
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #60a5fa;
      margin-bottom: 12px;
    }

    /* ── Dividers ── */
    [role="tabpanel"]:has(.or-playground) hr {
      border: none;
      height: 1px;
      margin: 24px 0;
      background: linear-gradient(90deg,
        transparent, rgba(255,255,255,0.1) 20%, rgba(255,255,255,0.1) 80%, transparent);
    }

    /* ── Sub-tab bar (Model View / Editor / Sweep) ── */
    [role="tabpanel"]:has(.or-playground) [data-baseweb="tab-list"] {
      background: rgba(255,255,255,0.05);
      border-radius: 12px;
      padding: 4px;
      border: 1px solid rgba(255,255,255,0.08);
      gap: 4px;
      margin-bottom: 16px;
    }
    [role="tabpanel"]:has(.or-playground) [data-baseweb="tab"] {
      font-size: 0.85rem;
      font-weight: 500;
      padding: 10px 20px;
      border-radius: 8px;
      color: #8896b0;
      transition: all 200ms ease;
    }
    [role="tabpanel"]:has(.or-playground) [data-baseweb="tab"]:hover {
      color: #c8d4e8;
      background: rgba(255,255,255,0.04);
    }
    [role="tabpanel"]:has(.or-playground) [data-baseweb="tab"][aria-selected="true"] {
      background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
      color: #fff !important;
      box-shadow: 0 3px 12px rgba(59,130,246,0.35);
    }

    /* ── Buttons ── */
    [role="tabpanel"]:has(.or-playground) .stButton > button,
    [role="tabpanel"]:has(.or-playground) .stFormSubmitButton > button {
      font-size: 0.875rem;
      font-weight: 500;
      padding: 10px 24px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.06);
      color: #c8d4e8;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      backdrop-filter: blur(8px);
      transition: all 180ms ease;
    }
    [role="tabpanel"]:has(.or-playground) .stButton > button:hover,
    [role="tabpanel"]:has(.or-playground) .stFormSubmitButton > button:hover {
      background: rgba(255,255,255,0.10);
      border-color: rgba(255,255,255,0.18);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
      color: #fff;
    }
    /* Primary button — blue gradient */
    [role="tabpanel"]:has(.or-playground) .stButton > button[kind="primary"],
    [role="tabpanel"]:has(.or-playground) .stFormSubmitButton > button {
      background: linear-gradient(135deg, #3b82f6, #2563eb);
      border: none;
      color: #fff;
      box-shadow: 0 4px 14px rgba(59,130,246,0.4);
    }
    [role="tabpanel"]:has(.or-playground) .stButton > button[kind="primary"]:hover,
    [role="tabpanel"]:has(.or-playground) .stFormSubmitButton > button:hover {
      background: linear-gradient(135deg, #60a5fa, #3b82f6);
      box-shadow: 0 6px 22px rgba(59,130,246,0.5);
      transform: translateY(-1px);
    }
    [role="tabpanel"]:has(.or-playground) .stButton > button:focus-visible {
      outline: none;
      box-shadow: 0 0 0 3px rgba(59,130,246,0.3);
    }

    /* ── Inputs (number, text, textarea) ── */
    [role="tabpanel"]:has(.or-playground) [data-testid="stNumberInput"] input,
    [role="tabpanel"]:has(.or-playground) [data-testid="stTextInput"] input,
    [role="tabpanel"]:has(.or-playground) [data-testid="stTextArea"] textarea {
      font-size: 0.9rem;
      padding: 10px 14px !important;
      border-radius: 10px !important;
      border: 1px solid rgba(255,255,255,0.10) !important;
      background: rgba(255,255,255,0.05) !important;
      color: #e0e7f0 !important;
      transition: all 180ms ease;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stNumberInput"] input:focus,
    [role="tabpanel"]:has(.or-playground) [data-testid="stTextInput"] input:focus {
      border-color: #3b82f6 !important;
      box-shadow: 0 0 0 3px rgba(59,130,246,0.2), 0 2px 8px rgba(0,0,0,0.2) !important;
      background: rgba(255,255,255,0.07) !important;
    }
    /* Input labels */
    [role="tabpanel"]:has(.or-playground) [data-testid="stNumberInput"] label,
    [role="tabpanel"]:has(.or-playground) [data-testid="stTextInput"] label,
    [role="tabpanel"]:has(.or-playground) [data-testid="stSelectbox"] label {
      font-size: 0.72rem !important;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #64748b !important;
    }
    /* Number stepper +/– */
    [role="tabpanel"]:has(.or-playground) [data-testid="stNumberInput"] button {
      border-radius: 8px !important;
      background: rgba(255,255,255,0.04) !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
      color: #8896b0 !important;
      transition: all 150ms ease;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stNumberInput"] button:hover {
      background: rgba(59,130,246,0.15) !important;
      border-color: rgba(59,130,246,0.4) !important;
      color: #60a5fa !important;
    }

    /* ── Selectbox ── */
    [role="tabpanel"]:has(.or-playground) [data-testid="stSelectbox"] [data-baseweb="select"] > div {
      border-radius: 10px !important;
      border: 1px solid rgba(255,255,255,0.10) !important;
      background: rgba(255,255,255,0.05) !important;
      transition: all 180ms ease;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
      border-color: #3b82f6 !important;
      box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
    }

    /* ── Expanders — frosted glass cards ── */
    [role="tabpanel"]:has(.or-playground) [data-testid="stExpander"] {
      background: rgba(255,255,255,0.04);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      overflow: hidden;
      margin-bottom: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
      transition: all 200ms ease;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stExpander"]:hover {
      border-color: rgba(255,255,255,0.14);
      box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stExpander"] summary {
      padding: 14px 18px;
      font-weight: 500;
      font-size: 0.92rem;
      color: #d0d8e8;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stExpander"] details[open] > summary {
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
      padding: 18px;
    }

    /* ── Forms — card treatment ── */
    [role="tabpanel"]:has(.or-playground) [data-testid="stForm"] {
      background: rgba(255,255,255,0.035);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }

    /* ── Metrics — result cards ── */
    [role="tabpanel"]:has(.or-playground) [data-testid="stMetric"] {
      background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.03) 100%);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 18px 22px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
      transition: all 200ms ease;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stMetric"]:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 28px rgba(0,0,0,0.3);
      border-color: rgba(255,255,255,0.14);
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stMetricLabel"] {
      font-size: 0.7rem !important;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #64748b !important;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stMetricValue"] {
      font-size: 2rem !important;
      font-weight: 700 !important;
      letter-spacing: -0.02em;
      color: #e8ecf4 !important;
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stMetricDelta"] {
      font-size: 0.85rem !important;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 999px;
      display: inline-block;
      margin-top: 6px;
    }

    /* ── Data tables / Charts ── */
    [role="tabpanel"]:has(.or-playground) [data-testid="stDataFrame"] {
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
    }
    [role="tabpanel"]:has(.or-playground) [data-testid="stPlotlyChart"] {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 8px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }

    /* ── Captions ── */
    [role="tabpanel"]:has(.or-playground) [data-testid="stCaptionContainer"] {
      font-size: 0.75rem !important;
      color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<p class="main-header">🎯 OR Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Operations Research Solutions</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Render solver settings
    from src.ui.solver_settings import render_solver_settings
    render_solver_settings()
    
    # MIPLIB Cache
    with st.expander("MIPLIB Cache"):
        from src.storage.miplib_cache import MIPLIBCache
        cache = MIPLIBCache()
        cached_instances = cache.list_cached()
        
        if cached_instances:
            import pandas as pd
            df = pd.DataFrame(cached_instances)
            
            # Format for compact display
            df['objective_value'] = df['objective_value'].apply(lambda x: f"{x:.2f}" if x else "N/A")
            df['solve_time'] = df['solve_time'].apply(lambda x: f"{x:.1f}s")
            
            # Extract just the date from timestamp
            df['timestamp'] = df['timestamp'].apply(lambda ts: ts.split('T')[0] if 'T' in ts else ts)
            
            # Select and rename columns
            df = df[['instance_name', 'model_id', 'objective_value', 'solver_used', 'timestamp']]
            df.columns = ['Instance', 'ID', 'Objective', 'Solver', 'Date']
            
            st.dataframe(df, width="stretch", hide_index=True, height=200)
            
            if st.button("Clear Cache", type="secondary", width="stretch"):
                cache.clear()
                st.rerun()
        else:
            st.info("No cached solutions")

    st.divider()

    # Example problems
    st.header("📚 Example Problems")
    example_category = st.selectbox(
        "Category",
        ["Linear Programming", "Integer Programming", "Transportation", "Assignment", "Scheduling"]
    )

    _EXAMPLES = {
        "Linear Programming": (
            "A furniture company makes tables (profit $50) and chairs "
            "(profit $30). Each table needs 3 hours labor and 1 unit "
            "material. Each chair needs 2 hours labor and 1 unit material. "
            "Total labor available: 240 hours. Total material available: "
            "100 units. How many of each should be produced to maximize profit?"
        ),
        "Integer Programming": (
            "A delivery company has 5 packages to assign to 3 trucks. "
            "Each truck can carry at most 100 kg. Package weights: "
            "[30, 45, 20, 55, 40]. Package values: [10, 15, 8, 20, 12]. "
            "Maximize total value delivered while respecting truck capacity."
        ),
        "Transportation": (
            "Minimize shipping costs from 3 warehouses to 4 stores. "
            "Warehouse capacities: [100, 150, 200]. "
            "Store demands: [80, 120, 90, 110]. "
            "Cost per unit shipped (rows=warehouses, cols=stores): "
            "[[5,8,6,7],[6,7,9,5],[8,6,7,9]]"
        ),
        "Assignment": (
            "Assign 4 workers to 4 tasks to minimize total cost. "
            "Cost matrix (rows=workers, cols=tasks): "
            "[[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]]"
        ),
        "Scheduling": (
            "Schedule 4 jobs on 2 machines. Job processing times on "
            "machine 1: [3,5,2,4]. On machine 2: [4,2,6,3]. "
            "Minimize total completion time."
        ),
    }

    if st.button("Load Example"):
        st.session_state.example_text = _EXAMPLES.get(example_category, "")
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💡 Solve Problem", "📊 Results", "📈 Visualization",
    "🧮 Model Playground", "🗄️ MIPLIB Cache", "📖 Help"
])

with tab1:
    st.header("Describe Your Problem")

    input_method = st.radio(
        "Input Method",
        ["Type Problem", "Upload File", "Load from MIPLIB"],
        horizontal=True,
    )

    # --- Shared variables set by any branch -----------------------------
    problem_description: str = ""
    _uploaded_file = None
    _file_hint: str = ""
    _miplib_instance: str = ""
    _miplib_load_clicked = False

    if input_method == "Upload File":
        _uploaded_file = st.file_uploader(
            "Upload your data file",
            type=["xlsx", "xls", "csv", "docx", "doc", "txt", "pdf", "mps", "gz"],
        )
        _file_hint = st.text_input(
            "Problem hint (optional)",
            placeholder="e.g. transportation problem, minimize cost",
        )

        if _uploaded_file is not None:
            with st.expander("📄 File Preview", expanded=True):
                _preview_bytes = _uploaded_file.getvalue()
                _ext = _uploaded_file.name.rsplit('.', 1)[-1].lower()
                if _ext in ('csv', 'xlsx', 'xls'):
                    try:
                        if _ext == 'csv':
                            import io as _io
                            _pv_df = pd.read_csv(_io.BytesIO(_preview_bytes))
                        else:
                            _pv_df = pd.read_excel(_io.BytesIO(_preview_bytes))
                        st.dataframe(_pv_df.head(10), width="stretch")
                        st.caption(f"{len(_pv_df)} rows × {len(_pv_df.columns)} columns")
                    except Exception as _prev_err:
                        st.text(f"Could not preview: {_prev_err}")
                else:
                    _raw_preview = _preview_bytes.decode('utf-8', errors='replace')[:500]
                    if len(_preview_bytes) > 500:
                        _raw_preview += "\n... [truncated]"
                    st.text(_raw_preview)

    elif input_method == "Load from MIPLIB":
        st.info(
            "**MIPLIB** is a library of 1,065 real-world optimization problems "
            "from industry and academia. Select a recommended instance or enter "
            "any MIPLIB instance name to download and solve it."
        )

        _rec_names = [f"{r['name']}  —  {r['desc']}" for r in RECOMMENDED_INSTANCES]
        _sel_rec = st.selectbox("Recommended instances", _rec_names)
        _rec_instance = _sel_rec.split("  —  ")[0].strip() if _sel_rec else ''

        _custom = st.text_input(
            "Or enter any MIPLIB instance name:",
            placeholder="e.g. neos-1067731",
        )
        _miplib_instance = _custom.strip() if _custom.strip() else _rec_instance

        _miplib_load_clicked = st.button(
            "📦 Load from MIPLIB", type="primary", width="stretch",
        )

        # --- Browse MIPLIB expander -------------------------------------
        with st.expander("🔍 Browse MIPLIB (easy instances)"):
            if 'miplib_easy_list' not in st.session_state:
                if st.button("Fetch instance list"):
                    try:
                        loader = MIPLIBLoader()
                        st.session_state.miplib_easy_list = loader.list_easy_instances()
                    except Exception as _e:
                        st.error(f"Could not fetch list: {_e}")

            if 'miplib_easy_list' in st.session_state:
                _easy = st.session_state.miplib_easy_list[:50]
                st.caption(f"Showing first 50 of {len(st.session_state.miplib_easy_list)} easy instances")
                for _inst_name in _easy:
                    _c1, _c2 = st.columns([3, 1])
                    _c1.markdown(f"`{_inst_name}`")
                    if _c2.button("Load", key=f"miplib_{_inst_name}"):
                        st.session_state.miplib_pick = _inst_name
                        st.rerun()

        _picked = st.session_state.pop('miplib_pick', None)
        if _picked:
            _miplib_instance = _picked
            _miplib_load_clicked = True  # auto-trigger the solve path

    else:
        problem_description = st.text_area(
            "Enter problem description in natural language:",
            value=st.session_state.get("example_text", ""),
            height=200,
            placeholder="Example: I need to minimize transportation costs between 3 warehouses and 5 stores. Warehouse 1 has 100 units available...",
            help="Describe your optimization problem. Be specific about objectives, constraints, and data.",
        )
        
        # Sync back to session_state and handle example_text cleanup
        if problem_description:
            # Clear example_text if user has edited the text
            if problem_description != st.session_state.get("example_text", ""):
                st.session_state.pop("example_text", None)
            # Store current text for fallback
            st.session_state.problem_description_text = problem_description

    col1, col2, _ = st.columns([1, 1, 2])

    with col1:
        solve_button = st.button("🚀 Solve Problem", type="primary", width="stretch")

    with col2:
        clear_button = st.button("🗑️ Clear", width="stretch")

    _pipeline_ready = False
    _mps_direct = False  # True when MPS data bypasses Step 2 AI generation

    # ====================================================================
    #  MIPLIB path
    # ====================================================================
    if _miplib_load_clicked and input_method == "Load from MIPLIB":
        if not _miplib_instance:
            st.warning("⚠️ Please select or enter a MIPLIB instance name.")
            st.stop()

        st.warning(
            "**Note:** MIPLIB problems are provided in mathematical form. "
            "The AI explanation will describe what was solved, but the "
            "problem description will be technical."
        )

        with st.status(
            f"Step 1: Loading MIPLIB instance '{_miplib_instance}'...",
            expanded=True,
        ) as s1:
            try:
                loader = MIPLIBLoader()

                with st.spinner(f"Downloading {_miplib_instance}.mps.gz..."):
                    parsed = loader.load_instance(_miplib_instance)
                st.session_state.file_data = parsed

                if parsed.get('type') == 'error':
                    s1.update(label="Step 1: MPS load failed", state="error")
                    st.error(f"❌ Failed to load: {parsed.get('error')}")
                    st.stop()

            except Exception as e:
                s1.update(label="Step 1: MIPLIB download failed", state="error")
                st.error(f"❌ MIPLIB download failed: {e}")
                st.stop()

            n_vars = parsed.get('num_variables', 0)
            n_cons = parsed.get('num_constraints', 0)
            obj_name = parsed.get('objective_name', '?')
            vbounds = parsed.get('variable_bounds', {})
            n_bin = sum(1 for vb in vbounds.values() if vb.get('type') == 'binary')
            n_int = sum(1 for vb in vbounds.values() if vb.get('type') == 'integer')
            n_cont = n_vars - n_bin - n_int

            # Preview card
            st.info(f"**Instance:** `{_miplib_instance}` — loaded from MIPLIB")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Variables", f"{n_vars:,}")
            mc2.metric("Constraints", f"{n_cons:,}")
            mc3.metric("Binary", f"{n_bin:,}")
            mc4.metric("Integer", f"{n_int:,}")

            if n_bin or n_int:
                _ptype_badge = "Mixed Integer Programming"
            else:
                _ptype_badge = "Linear Programming"
            st.markdown(f"**Problem type:** `{_ptype_badge}`")
            _miplib_obj_sense = parsed.get('objective_sense', 'minimize')
            st.markdown(f"**Objective:** {_miplib_obj_sense} `{obj_name}`")

            # Try fetching known optimal value
            try:
                _info = loader.get_instance_info(_miplib_instance)
                _best = _info.get('best_known_objective')
                if _best is not None:
                    st.markdown(f"**Known optimal value:** `{_best}`")
            except Exception:
                pass

            # Build a problem_data dict compatible with the rest of the pipeline
            problem_data = {
                'problem_type': 'mixed_integer_programming' if (n_bin or n_int) else 'linear_programming',
                'objective': _miplib_obj_sense,
                'objective_description': f'{_miplib_obj_sense.capitalize()} {obj_name} (MIPLIB benchmark)',
                'decision_variables': [
                    {
                        'name': v,
                        'type': vbounds.get(v, {}).get('type', 'continuous'),
                        'description': '',
                    }
                    for v in parsed.get('variables', [])[:50]
                ],
                'constraints': [],
                'parameters': {},
                'confidence': 1.0,
                'assumptions': [],
                'warnings': [],
                'notes': f'MIPLIB instance {_miplib_instance}',
                'source': 'miplib',
                'filename': f'{_miplib_instance}.mps.gz',
                'instance_name': _miplib_instance,
            }
            st.session_state.problem_data = problem_data
            st.session_state.mps_parsed = parsed

            s1.update(
                label=(
                    f"Step 1 Done: {_miplib_instance} — "
                    f"{n_vars:,} vars, {n_cons:,} constraints"
                ),
                state="complete",
            )

        _pipeline_ready = True
        _mps_direct = True

    # ====================================================================
    #  File-upload path
    # ====================================================================
    elif solve_button and input_method == "Upload File":
        if _uploaded_file is None:
            st.warning("⚠️ Please upload a file first.")
            st.stop()

        # --- Step 1 (file): Parse & extract -----------------------------
        with st.status("Step 1: Analysing uploaded file...", expanded=True) as s1:
            try:
                parser = FileParser()
                parsed = parser.parse(
                    _uploaded_file.getvalue(),
                    filename=_uploaded_file.name,
                )
                st.session_state.file_data = parsed

                if parsed.get('type') == 'error':
                    s1.update(label="Step 1: File parsing failed", state="error")
                    st.error(f"❌ File parsing failed: {parsed.get('error')}")
                    st.stop()

                # MPS files are already fully parsed — bypass AI DataExtractor entirely.
                # DataExtractor ignores structured MPS data and its defaults hardcode 'minimize'.
                if parsed.get('type') == 'mps':
                    parsed['source'] = 'mps_upload'  # kept by generate_from_mps()
                    n_vars = parsed.get('num_variables', 0)
                    n_cons = parsed.get('num_constraints', 0)
                    vbounds = parsed.get('variable_bounds', {})
                    n_bin = sum(1 for vb in vbounds.values() if vb.get('type') == 'binary')
                    n_int = sum(1 for vb in vbounds.values() if vb.get('type') == 'integer')
                    _obj_sense = parsed.get('objective_sense', 'minimize')

                    problem_data = {
                        'problem_type': 'mixed_integer_programming' if (n_bin or n_int) else 'linear_programming',
                        'objective': _obj_sense,
                        'objective_description': (
                            f"{_obj_sense.capitalize()} {parsed.get('objective_name', 'objective')} "
                            f"(uploaded MPS)"
                        ),
                        'decision_variables': [
                            {
                                'name': v,
                                'type': vbounds.get(v, {}).get('type', 'continuous'),
                                'description': '',
                            }
                            for v in parsed.get('variables', [])[:50]
                        ],
                        'constraints': [],
                        'parameters': {},
                        'confidence': 1.0,
                        'assumptions': [],
                        'warnings': [],
                        'notes': f'Uploaded MPS file: {_uploaded_file.name}',
                        'source': 'mps_upload',
                        'filename': _uploaded_file.name,
                    }
                    st.session_state.mps_parsed = parsed
                    _mps_direct = True
                else:
                    extractor = DataExtractor()
                    problem_data = extractor.extract(
                        parsed,
                        user_hint=_file_hint,
                        filename=_uploaded_file.name,
                    )

                st.session_state.problem_data = problem_data
            except Exception as e:
                s1.update(label="Step 1: File analysis failed", state="error")
                st.error(f"❌ File analysis failed: {e}")
                st.stop()

            conf = problem_data.get('confidence', 0)
            ptype = problem_data.get('problem_type', 'N/A')

            st.info(f"File analysed: **{_uploaded_file.name}** — detected **{ptype}**")
            st.markdown(f"**Objective:** `{problem_data.get('objective', '?')}`")
            st.markdown(f"**Confidence:** `{conf:.0%}`")

            # Show rich metrics for MPS uploads just like MIPLIB path does
            if _mps_direct:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Variables", f"{n_vars:,}")
                mc2.metric("Constraints", f"{n_cons:,}")
                mc3.metric("Binary", f"{n_bin:,}")
                mc4.metric("Integer", f"{n_int - n_bin:,}")

            ds = problem_data.get('data_summary', {})
            if ds:
                st.markdown(
                    f"**Data:** {ds.get('num_rows', '?')} rows × "
                    f"{ds.get('num_columns', '?')} columns — "
                    f"{ds.get('detected_structure', '')}"
                )

            dec_vars = problem_data.get('decision_variables', [])
            if dec_vars:
                st.markdown(f"**Decision Variables** ({len(dec_vars)} found):")
                for v in dec_vars:
                    name = v.get('name', '?')
                    vtype = v.get('type', 'continuous')
                    desc = v.get('description', '')
                    st.markdown(f"- `{name}` ({vtype}): {desc}")

            constraints = problem_data.get('constraints', [])
            if constraints:
                st.markdown(f"**Constraints** ({len(constraints)} found):")
                for c in constraints:
                    if isinstance(c, dict):
                        cname = c.get('name', c.get('description', ''))
                        expr = c.get('expression', c.get('formula', ''))
                        st.markdown(f"- `{cname}`: `{expr}`" if expr else f"- {cname}")
                    else:
                        st.markdown(f"- `{c}`")

            pd_warnings = problem_data.get('warnings', [])
            if pd_warnings:
                st.markdown(f"**Warnings:** {', '.join(str(w) for w in pd_warnings)}")

            s1.update(
                label=f"Step 1 Done: {ptype} ({conf:.0%} confidence)",
                state="complete",
            )

        st.metric("Confidence", f"{conf:.0%}")

        if conf < 0.15:
            warns = "\n".join(str(w) for w in problem_data.get('warnings', []))
            st.error(
                f"⚠️ Could not identify an OR problem in this file "
                f"(confidence: {conf:.0%}).\n\nWarnings: {warns}"
            )
            st.stop()

        _pipeline_ready = True

    # ====================================================================
    #  Text-input path
    # ====================================================================
    # Fallback to session_state if widget value was lost
    if solve_button and not problem_description:
        problem_description = st.session_state.get("problem_description_text", "")
    
    elif solve_button and problem_description:
        # --- Input validation -------------------------------------------
        _stripped = problem_description.strip()
        if len(_stripped) < 20:
            st.warning(
                "⚠️ Please describe your problem in more detail. "
                "Include what you want to optimize, your variables, "
                "and your constraints."
            )
            st.stop()

        if _stripped.lower() in (
            "clear", "test", "hello", "hi", "help", "?", "ok",
        ):
            st.error(
                "This doesn't look like an OR problem. "
                "Try using the **Load Example** button to see the expected format."
            )
            st.stop()

        # --- Step 1: Classify -------------------------------------------
        with st.status("Step 1: Classifying problem...", expanded=True) as s1:
            try:
                classifier = ProblemClassifier()
                problem_data = classifier.classify(problem_description)
                st.session_state.problem_data = problem_data
            except Exception as e:
                s1.update(label="Step 1: Classification failed", state="error")
                st.error(f"❌ Classification failed: {e}")
                st.stop()

            conf = problem_data.get('confidence', 0)
            ptype = problem_data.get('problem_type', 'N/A')
            objective = problem_data.get('objective', '?')

            st.markdown(f"**Problem type:** `{ptype}`")
            st.markdown(f"**Objective:** `{objective}`")
            st.markdown(f"**Confidence:** `{conf:.0%}`")

            dec_vars = problem_data.get('decision_variables', [])
            if dec_vars:
                st.markdown(f"**Decision Variables** ({len(dec_vars)} found):")
                for v in dec_vars:
                    name = v.get('name', '?')
                    vtype = v.get('type', 'continuous')
                    desc = v.get('description', '')
                    st.markdown(f"- `{name}` ({vtype}): {desc}")

            constraints = problem_data.get('constraints', [])
            if constraints:
                st.markdown(f"**Constraints** ({len(constraints)} found):")
                for c in constraints:
                    if isinstance(c, dict):
                        cname = c.get('name', c.get('description', ''))
                        expr = c.get('expression', c.get('formula', ''))
                        st.markdown(f"- `{cname}`: `{expr}`" if expr else f"- {cname}")
                    else:
                        st.markdown(f"- `{c}`")

            pd_warnings = problem_data.get('warnings', [])
            if pd_warnings:
                st.markdown(f"**Warnings:** {', '.join(pd_warnings)}")

            s1.update(
                label=f"Step 1 Done: {ptype} ({conf:.0%} confidence)",
                state="complete",
            )

        st.metric("Confidence", f"{conf:.0%}")

        if conf < 0.15:
            warns = "\n".join(problem_data.get('warnings', []))
            st.error(
                f"⚠️ Could not identify this as an OR problem "
                f"(confidence: {conf:.0%}).\n\nWarnings: {warns}"
            )
            st.stop()

        _pipeline_ready = True

    elif solve_button:
        st.warning("⚠️ Please enter a problem description or upload a file first.")

    if clear_button:
        problem_description = ""
        st.session_state.pop("example_text", None)
        st.session_state.pop("problem_description_text", None)
        st.rerun()

    # ====================================================================
    #  Shared pipeline — Steps 2-4 (runs for all input paths)
    # ====================================================================
    if _pipeline_ready:
        problem_data = st.session_state.problem_data

        # --- Step 2: Build model ----------------------------------------
        from src.ui.solver_settings import get_selected_solver_key
        solver_key = get_selected_solver_key()
        _solver_pref = 'cvxpy' if solver_key.startswith('cvxpy') else 'pulp'
        with st.status("Step 2: Building mathematical model...", expanded=True) as s2:
            try:
                generator = ModelGenerator()
                if _mps_direct:
                    model, _mps_pdata = generator.generate_from_mps(
                        st.session_state.mps_parsed,
                        solver_preference=_solver_pref,
                    )
                    st.session_state.problem_data.update(_mps_pdata)
                    problem_data = st.session_state.problem_data
                else:
                    model = generator.generate(
                        problem_data,
                        solver_preference=_solver_pref,
                    )
                st.session_state.model = model
            except Exception as e:
                s2.update(label="Step 2: Model generation failed", state="error")
                st.error(f"❌ Model generation failed: {e}")
                st.stop()

            if isinstance(model, dict) and model.get('type') == 'cvxpy':
                cp_prob = model['problem']
                import cvxpy as _cp
                if isinstance(cp_prob, _cp.Problem):
                    n_vars = len(model.get('variables', {}))
                    n_cons = len(cp_prob.constraints)
                else:
                    n_vars = len(cp_prob.variables())
                    n_cons = len(cp_prob.constraints)
            else:
                n_vars = len(model.variables())
                n_cons = len(model.constraints)

            st.markdown(f"**Variables:** {n_vars}  &nbsp;|&nbsp;  **Constraints:** {n_cons}")

            if isinstance(model, dict) and model.get('type') == 'cvxpy':
                import cvxpy as _cp
                _inner = model.get('problem')
                if isinstance(_inner, _cp.Problem):
                    st.markdown(f"**Objective:** `{_inner.objective}`")
                    if _inner.constraints:
                        st.markdown("**Constraints:**")
                        for i, c in enumerate(_inner.constraints):
                            st.markdown(f"- constraint_{i}: `{c}`")
                    st.markdown("**Variables:**")
                    for vname in model.get('variables', {}):
                        st.markdown(f"- `{vname}`")
                else:
                    st.markdown("*(PuLP model will be auto-converted to CVXPY at solve time)*")
            else:
                import pulp as _pulp
                sense_label = "Maximize" if model.sense == _pulp.constants.LpMaximize else "Minimize"
                st.markdown(f"**Objective ({sense_label}):**")
                if _mps_direct:
                    st.markdown(f"`{model.objective}`")
                else:
                    st.code(str(model.objective), language=None)

                if model.constraints:
                    if _mps_direct and len(model.constraints) > 20:
                        st.markdown(f"**Constraints:** {len(model.constraints)} (showing first 20)")
                        for i, (cname, cobj) in enumerate(model.constraints.items()):
                            if i >= 20:
                                break
                            cstr = str(cobj).replace("-0.0", "0").replace("+ -", "- ")
                            st.markdown(f"- `{cname}`:  `{cstr}`")
                    else:
                        st.markdown("**Constraint formulas:**")
                        for cname, cobj in model.constraints.items():
                            cstr = str(cobj).replace("-0.0", "0").replace("+ -", "- ")
                            st.markdown(f"- `{cname}`:  `{cstr}`")

                if _mps_direct and len(model.variables()) > 20:
                    st.markdown(f"**Variables:** {len(model.variables())} (showing first 20)")
                    for v in model.variables()[:20]:
                        cat = v.cat
                        lo = v.lowBound
                        hi = v.upBound
                        st.markdown(f"- `{v.name}` ({cat}, bounds [{lo}, {hi}])")
                else:
                    st.markdown("**Variable details:**")
                    for v in model.variables():
                        cat = v.cat
                        lo = v.lowBound
                        hi = v.upBound
                        bounds = f"[{lo}, {hi}]"
                        st.markdown(f"- `{v.name}` ({cat}, bounds {bounds})")

            s2.update(
                label=f"Step 2 Done: {n_vars} variables, {n_cons} constraints",
                state="complete",
            )

        # --- Step 3: Solve ----------------------------------------------
        from src.ui.solver_settings import get_selected_solver_key, get_selected_time_limit
        from src.ui.solver_progress import show_pre_solve_info, show_solve_result, show_cached_result_banner, show_variable_values
        from src.solvers.solver_router import resolve_solver
        
        # Get solver configuration
        solver_key = get_selected_solver_key()
        max_time = get_selected_time_limit()
        
        # Resolve the actual solver to use
        resolved_key, resolver_explanation = resolve_solver(solver_key, st.session_state.problem_data)
        
        # Solve the problem
        with st.status("Step 3: Solving...", expanded=True) as s3:
            try:
                # Create solver interface
                solver = SolverInterface(
                    solver_type=resolved_key,
                    problem_data=st.session_state.problem_data
                )
                
                # Check if solution will come from cache
                if solver.will_use_cache:
                    # Solve (will return cached result)
                    solution = solver.solve(model, max_time=max_time)
                    st.session_state.solution = solution
                    
                    # Show cached result banner
                    show_cached_result_banner(solution)
                else:
                    # Get problem dimensions
                    if isinstance(model, dict) and model.get('type') == 'cvxpy':
                        cvxpy_prob = model.get('problem')
                        num_vars = len(cvxpy_prob.variables()) if cvxpy_prob else 0
                        num_constraints = len(cvxpy_prob.constraints) if cvxpy_prob else 0
                    else:
                        num_vars = len(model.variables()) if hasattr(model, 'variables') else 0
                        num_constraints = len(model.constraints) if hasattr(model, 'constraints') else 0
                    
                    # Show pre-solve info
                    timer_placeholder = show_pre_solve_info(
                        resolved_key, resolver_explanation, num_vars, num_constraints, max_time, st.session_state.problem_data
                    )
                    
                    # Solve
                    # For MIPLIB with SCIP, show a spinner since it can take a while
                    if st.session_state.problem_data.get('source') == 'miplib' and 'scip' in resolved_key.lower():
                        with st.spinner(f"⏳ SCIP is solving the problem (up to {max_time}s)... Please wait."):
                            solution = solver.solve(model, max_time=max_time)
                    else:
                        solution = solver.solve(model, max_time=max_time)
                    st.session_state.solution = solution
                    
                    # Clear timer and show result
                    timer_placeholder.empty()
                    show_solve_result(solution)
                    
                    # Show variable values
                    show_variable_values(solution)
                
                # Update status
                s3.update(label="Step 3: Complete", state="complete")
                
                # Handle sensitivity analysis (if optimal)
                if solution.get('is_optimal'):
                    try:
                        st.session_state.sensitivity = solver.get_sensitivity(model)
                    except:
                        st.session_state.sensitivity = None
                else:
                    st.session_state.sensitivity = None
                    
            except Exception as e:
                s3.update(label="Step 3: Failed", state="error")
                st.error(f"Solver failed: {e}")
                st.stop()

        # --- Step 4: Interpret ------------------------------------------
        with st.status("Step 4: Generating AI explanation...", expanded=True) as s4:
            try:
                interpreter = ResultInterpreter()
                interpretation = interpreter.interpret(solution, problem_data)
                st.session_state.interpretation = interpretation
            except Exception as e:
                s4.update(label="Step 4: Interpretation failed", state="error")
                st.error(f"❌ Interpretation failed: {e}")
                st.stop()

            st.markdown(f"**Summary:** {interpretation.get('summary', '')}")
            findings = interpretation.get('key_findings', [])
            if findings:
                st.markdown("**Key findings:**")
                for f in findings:
                    st.markdown(f"- {f}")

            s4.update(label="Step 4 Done: Interpretation ready", state="complete")

        st.session_state.solution_ready = True

        if 'playground_original_state' not in st.session_state:
            _m = st.session_state.get('model')
            if _m is not None and hasattr(_m, 'variables'):
                from src.utils.model_editor import pulp_to_editor_state
                st.session_state.playground_original_state = pulp_to_editor_state(_m)
                st.session_state.playground_original_solution = deepcopy(solution)

        st.balloons()

with tab2:
    st.header("Solution Results")

    if 'solution' in st.session_state:
        sol = st.session_state.solution
        interp = st.session_state.get('interpretation', {})
        pdata = st.session_state.get('problem_data', {})

        # ---- Metric cards ------------------------------------------------
        m1, m2, m3, m4 = st.columns(4)

        obj_val = sol.get('objective_value')
        cost_types = ('transportation', 'assignment', 'scheduling')
        if pdata.get('problem_type', '') in cost_types and obj_val is not None:
            obj_display = f"${obj_val:,.2f}"
        elif obj_val is not None:
            obj_display = f"{obj_val:,.4f}"
        else:
            obj_display = "N/A"
        m1.metric("Objective Value", obj_display)

        is_opt = sol.get('is_optimal', False)
        m2.metric(
            "Solution Status",
            "Optimal ✅" if is_opt else f"{sol.get('status', 'Unknown')} ❌",
        )

        t = sol.get('solve_time')
        m3.metric("Solve Time", f"{t:.3f}s" if t is not None else "N/A")

        m4.metric("Variables", str(sol.get('num_variables', '?')))

        st.divider()

        # ---- AI Explanation ----------------------------------------------
        st.subheader("🤖 AI Explanation")

        summary = interp.get('summary', '')
        if summary:
            st.info(summary)

        findings = interp.get('key_findings', [])
        if findings:
            st.markdown("**Key Findings:**")
            for finding in findings:
                st.markdown(f"- {finding}")

        st.divider()

        # ---- Recommendations --------------------------------------------
        recs = interp.get('recommendations', [])
        if recs:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(recs, 1):
                st.markdown(f"{i}. {rec}")

        # ---- Warnings ----------------------------------------------------
        warns = interp.get('warnings', [])
        if warns:
            for w in warns:
                st.warning(w)

        # ---- Business Impact ---------------------------------------------
        impact = interp.get('business_impact', '')
        if impact:
            st.divider()
            st.subheader("📈 Business Impact")
            st.markdown(impact)

        st.divider()

        # ---- Decision Variables table ------------------------------------
        st.subheader("📋 Decision Variables")
        variables = sol.get('variables', {})
        if variables:
            df_vars = pd.DataFrame([
                {
                    "Variable": k,
                    "Value": v if v is not None else 0.0,
                    "Non-Zero": abs(v) > 0.001 if v is not None else False,
                }
                for k, v in variables.items()
            ])
            st.dataframe(df_vars, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Results (CSV)",
                df_vars.to_csv(index=False),
                "results.csv",
                "text/csv",
            )
        else:
            st.info("No variable data available.")

        # ---- Sensitivity Analysis table ----------------------------------
        sensitivity = st.session_state.get('sensitivity')
        if sensitivity and sensitivity.get('available'):
            st.divider()
            st.subheader("📊 Sensitivity Analysis — Shadow Prices")

            shadow = sensitivity.get('shadow_prices', {})
            if shadow:
                df_shadow = pd.DataFrame([
                    {
                        "Constraint": name,
                        "Shadow Price": round(price, 6),
                        "Interpretation": (
                            "Binding — each unit of relaxation changes "
                            f"objective by {abs(price):.4f}"
                            if abs(price) > 1e-8
                            else "Non-binding — slack available"
                        ),
                    }
                    for name, price in shadow.items()
                ])
                st.dataframe(df_shadow, width="stretch", hide_index=True)

            reduced = sensitivity.get('reduced_costs', {})
            if reduced:
                st.subheader("📉 Reduced Costs")
                df_rc = pd.DataFrame([
                    {
                        "Variable": name,
                        "Reduced Cost": round(rc, 6) if rc is not None else 0.0,
                    }
                    for name, rc in reduced.items()
                ])
                st.dataframe(df_rc, width="stretch", hide_index=True)
    else:
        st.info("👈 Solve a problem to see results here")

with tab3:
    st.header("📈 Visualizations")
    
    if 'solution' not in st.session_state:
        st.info("👈 Solve a problem first to see visualizations")
    else:
        from src.visualization.chart_generator import ChartGenerator
        _cg = ChartGenerator()
        _dark = dict(paper_bgcolor='#1E1E2E', plot_bgcolor='#1E1E2E', font=dict(color='white'))
        
        _sol = st.session_state.solution
        _pdata = st.session_state.get('problem_data') or {}
        _ptype = _pdata.get('problem_type', '')
        _vars = _sol.get('variables') or {}
        
        # Build non-zero variable list (used by most charts)
        _nz = [(k, v) for k, v in _vars.items() if v is not None and abs(v) > 0.001]
        _nz.sort(key=lambda x: abs(x[1]), reverse=True)
        _n_nz = len(_nz)
        
        # --- chart menu (universal) --------------------------------------
        chart_names = [
            'Variable Values (Bar)',
            'Top Variables (Horizontal)',
            'Variable Distribution (Pie)',
            'Variable Groups',
            'Solution Summary',
        ]
        
        # Add sensitivity only when data exists
        _sens = st.session_state.get('sensitivity')
        if isinstance(_sens, dict) and _sens.get('shadow_prices'):
            chart_names.append('Sensitivity Analysis')
        
        chart_sel = st.selectbox('Select Chart Type', chart_names)
        
        fig = None  # every branch below must set this
        
        # =================================================================
        #  1. Variable Values — vertical bar chart
        # =================================================================
        if chart_sel == 'Variable Values (Bar)':
            if _nz:
                _top = _nz[:50]
                _df = pd.DataFrame(_top, columns=['Variable', 'Value'])
                fig = px.bar(
                    _df, x='Variable', y='Value',
                    title=f'Non-zero Variable Values (showing {len(_df)} of {_n_nz})',
                    color='Value', color_continuous_scale='Viridis',
                )
                fig.update_layout(xaxis_tickangle=-45, height=550, **_dark)
        
        # =================================================================
        #  2. Top Variables — horizontal sorted bar
        # =================================================================
        elif chart_sel == 'Top Variables (Horizontal)':
            if _nz:
                # Show top-15 and bottom-15 when >30 non-zero vars
                _show = _nz[:30] if _n_nz <= 30 else (_nz[:15] + _nz[-15:])
                _show.sort(key=lambda x: x[1])
                _df = pd.DataFrame(_show, columns=['Variable', 'Value'])
                fig = px.bar(
                    _df, y='Variable', x='Value',
                    title=f'Variable Values — Sorted ({len(_df)} shown)',
                    orientation='h', color='Value',
                    color_continuous_scale='Viridis',
                )
                fig.update_layout(height=max(450, len(_df) * 22), **_dark)
        
        # =================================================================
        #  3. Variable Distribution — pie / donut
        # =================================================================
        elif chart_sel == 'Variable Distribution (Pie)':
            if _nz:
                _top10 = _nz[:10]
                _rest = sum(abs(v) for _, v in _nz[10:])
                labels = [k for k, _ in _top10]
                values = [abs(v) for _, v in _top10]
                if _rest > 0:
                    labels.append(f'Others ({_n_nz - 10})')
                    values.append(_rest)
                fig = go.Figure(go.Pie(
                    labels=labels, values=values, hole=0.35,
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#1E1E2E', width=2)),
                ))
                fig.update_layout(title='Variable Contribution to Objective', height=550, **_dark)
        
        # =================================================================
        #  4. Variable Groups — grouped by name prefix
        # =================================================================
        elif chart_sel == 'Variable Groups':
            groups: dict = {}
            for var, val in _nz:
                m = _re.match(r'^([a-zA-Z_]+)', var)
                grp = m.group(1).rstrip('_') if m else 'numeric'
                if grp not in groups:
                    groups[grp] = {'count': 0, 'total': 0.0}
                groups[grp]['count'] += 1
                groups[grp]['total'] += abs(val)
            
            if groups:
                _gdf = pd.DataFrame([
                    {'Group': g, 'Count': d['count'], 'Total |Value|': round(d['total'], 4)}
                    for g, d in sorted(groups.items(), key=lambda x: -x[1]['total'])
                ])
                fig = go.Figure()
                fig.add_trace(go.Bar(x=_gdf['Group'], y=_gdf['Count'], name='# Variables'))
                fig.add_trace(go.Bar(x=_gdf['Group'], y=_gdf['Total |Value|'], name='Sum |Value|', yaxis='y2'))
                fig.update_layout(
                    title=f'Variables by Name Prefix ({len(groups)} groups, {_n_nz} non-zero)',
                    yaxis=dict(title='Count'),
                    yaxis2=dict(title='Sum |Value|', overlaying='y', side='right'),
                    barmode='group', height=500, **_dark,
                )
        
        # =================================================================
        #  5. Solution Summary — key metrics table
        # =================================================================
        elif chart_sel == 'Solution Summary':
            _total = len(_vars)
            _bin = sum(1 for v in _vars.values() if v in (0, 1, 0.0, 1.0))
            obj = _sol.get('objective_value')
            obj_str = f'{obj:,.6f}' if obj is not None else 'N/A'
            
            rows = [
                ['Total Variables', f'{_total:,}'],
                ['Non-zero Variables', f'{_n_nz:,}'],
                ['Zero Variables', f'{_total - _n_nz:,}'],
                ['Binary (0/1) Variables', f'{_bin:,}'],
                ['Objective Value', obj_str],
                ['Status', _sol.get('status', 'Unknown')],
                ['Solve Time', f"{_sol.get('solve_time', 0):.2f} s"],
                ['Solver', _sol.get('solver_name', 'Unknown')],
                ['Problem Type', _ptype or 'Unknown'],
            ]
            
            fig = go.Figure(go.Table(
                header=dict(values=['Metric', 'Value'],
                            fill_color='#2E2E3E',
                            font=dict(color='white', size=14), align='left'),
                cells=dict(values=list(zip(*rows)),
                           fill_color='#1E1E2E',
                           font=dict(color='white', size=13), align='left', height=30),
            ))
            fig.update_layout(title='Solution Summary', height=420, **_dark)
        
        # =================================================================
        #  6. Sensitivity Analysis (only in menu when data exists)
        # =================================================================
        elif chart_sel == 'Sensitivity Analysis':
            shadow = _sens.get('shadow_prices', {}) if isinstance(_sens, dict) else {}
            nz_shadow = {k: v for k, v in shadow.items() if abs(v) > 1e-8}
            if nz_shadow:
                names = list(nz_shadow.keys())
                vals = list(nz_shadow.values())
                colors = ['#e74c3c' if abs(v) > 0.001 else '#7f8c8d' for v in vals]
                fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors))
                fig.add_hline(y=0, line_color='gray', line_width=1)
                fig.update_layout(
                    title='Shadow Prices (Binding Constraints)',
                    xaxis_title='Constraint', yaxis_title='Shadow Price',
                    height=500, **_dark,
                )
        
        # --- render chart or show helpful message -------------------------
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
            
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                st.download_button(
                    '📥 Download Chart (HTML)',
                    data=fig.to_html(include_plotlyjs='cdn'),
                    file_name=f'or_chart_{_ptype or "solution"}.html',
                    mime='text/html',
                )
        elif _n_nz == 0:
            st.info('All variable values are zero — nothing to chart.')
        else:
            st.warning('Could not generate this chart. Try a different chart type.')

# Tabs 5-6 and the footer are rendered before the Model Playground: its
# guard clauses call st.stop(), which halts the whole script, and would
# otherwise leave these tabs blank. Tab order on screen comes from st.tabs().
with tab5:
    st.header("🗄️ MIPLIB Cache")
    
    from src.storage.miplib_cache import MIPLIBCache
    cache = MIPLIBCache()
    
    # Cache statistics
    cache_stats = cache.get_cache_size()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cached Problems", cache_stats['num_entries'])
    with col2:
        st.metric("Cache Size", f"{cache_stats['size_mb']} MB")
    with col3:
        if cache_stats['newest_entry']:
            from datetime import datetime
            try:
                newest = datetime.fromisoformat(cache_stats['newest_entry'])
                st.metric("Last Cached", newest.strftime("%Y-%m-%d"))
            except:
                st.metric("Last Cached", "N/A")
        else:
            st.metric("Last Cached", "Empty")
    
    # List cached instances
    cached_instances = cache.list_cached()
    
    if cached_instances:
        st.subheader("Cached MIPLIB Instances")
        st.markdown(f"*Showing {len(cached_instances)} cached solutions*")
        
        # Create a dataframe for display
        import pandas as pd
        df = pd.DataFrame(cached_instances)
        
        # Format the dataframe
        df['solve_time'] = df['solve_time'].apply(lambda x: f"{x:.2f}s")
        df['objective_value'] = df['objective_value'].apply(lambda x: f"{x:.6f}" if x else "N/A")
        
        # Format timestamp
        def format_timestamp(ts):
            try:
                dt = datetime.fromisoformat(ts)
                return dt.strftime("%Y-%m-%d %H:%M")
            except:
                return ts
        
        df['timestamp'] = df['timestamp'].apply(format_timestamp)
        
        # Rename columns for display
        df = df.rename(columns={
            'instance_name': 'Instance',
            'model_id': 'Model ID',
            'objective_value': 'Objective Value',
            'solver_used': 'Solver',
            'solve_time': 'Solve Time',
            'timestamp': 'Cached At'
        })
        
        # Display the table
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Instance": st.column_config.TextColumn(width="medium"),
                "Model ID": st.column_config.TextColumn(
                    width="small", help="Hash of the model's content"
                ),
                "Objective Value": st.column_config.TextColumn(width="medium"),
                "Solver": st.column_config.TextColumn(width="small"),
                "Solve Time": st.column_config.TextColumn(width="small"),
                "Cached At": st.column_config.TextColumn(width="medium"),
            }
        )
        
        # Action buttons
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear All Cache", type="secondary"):
                cache.clear()
                st.success("Cache cleared!")
                st.rerun()
    else:
        st.info("No cached MIPLIB instances yet. Solve a MIPLIB problem to see cached results here.")

with tab6:
    st.header("How to Use OR Assistant")

    st.markdown("""
    ### 🎯 Getting Started

    1. **Describe Your Problem**: Type a natural language description, **upload a data file** (Excel, CSV, Word, PDF, TXT, MPS), or **load a benchmark** from MIPLIB
    2. **Select Solver**: Choose your preferred solver (or use auto-detect)
    3. **Click Solve**: Let the AI understand, model, and solve your problem
    4. **Review Results**: Get actionable insights and visualizations

    ### 📝 Problem Types Supported

    - **Linear Programming (LP)**: Optimize linear objectives with linear constraints
    - **Integer Programming (IP)**: Optimization with discrete decision variables
    - **Transportation**: Minimize costs of shipping goods from sources to destinations
    - **Assignment**: Optimally match resources to tasks
    - **Scheduling**: Sequence tasks over time to meet deadlines

    ### 💡 Example Problems

    **Transportation Problem:**
    ```
    I need to minimize transportation costs between 3 warehouses and 5 stores.
    Warehouse capacities: [100, 150, 120]
    Store demands: [80, 90, 60, 70, 50]
    Costs per unit: [cost matrix here]
    ```

    **Production Planning:**
    ```
    Maximize profit from producing 3 products using 2 machines.
    Machine 1 has 480 hours available, Machine 2 has 380 hours.
    Product A: 8 hours on M1, 4 hours on M2, profit $50
    [continue with other products...]
    ```

    ### ⚙️ Tips for Best Results

    - Be specific about constraints and objectives
    - Include all numerical data
    - Clearly state what you want to minimize or maximize
    - Mention any special requirements (integer values, time windows, etc.)

    ### 🆘 Need Help?

    - Check the [Documentation](docs/USER_GUIDE.md)
    - See [Examples](data/examples/)
    - Open an issue on GitHub
    """)

# Footer
st.divider()
st.caption("OR Assistant v0.1.0 | Built with AI & Streamlit | NDSU Project")


# =====================================================================
#  TAB 4 — Model Playground
# =====================================================================
with tab4:
    st.header("🧮 Model Playground")
    st.markdown('<div class="or-playground"></div>', unsafe_allow_html=True)

    from src.utils.model_editor import (
        pulp_to_editor_state,
        editor_state_to_pulp,
        diff_states,
        get_or_template,
        latex_expr_to_coefficients,
        parse_constraint_latex,
    )
    from src.ui.mathlive_component import mathlive_editor

    # ------------------------------------------------------------------
    #  Guard clause — no model solved yet
    # ------------------------------------------------------------------
    if 'playground_original_state' not in st.session_state:
        st.info(
            "**No model loaded yet.** Solve a problem in the *Solve Problem* tab first, "
            "or pick a template below to start from scratch."
        )
        st.markdown("#### Quick-start Templates")
        _tpl_cols = st.columns(5)
        _tpl_names = [
            ("Linear Program", "lp"),
            ("Integer Program", "ip"),
            ("Transportation", "transportation"),
            ("Knapsack", "knapsack"),
            ("Assignment", "assignment"),
        ]
        for _col, (_label, _tkey) in zip(_tpl_cols, _tpl_names):
            with _col:
                if st.button(_label, key=f"tpl_{_tkey}", width="stretch"):
                    st.session_state.playground_original_state = get_or_template(_tkey)
                    st.session_state.playground_original_solution = None
                    st.rerun()

        st.stop()

    # ------------------------------------------------------------------
    #  Initialise playground editor state from original if not yet done
    # ------------------------------------------------------------------
    if 'playground_editor_state' not in st.session_state:
        st.session_state.playground_editor_state = deepcopy(
            st.session_state.playground_original_state
        )

    _pg_orig = st.session_state.playground_original_state
    _pg_edit = st.session_state.playground_editor_state

    # ------------------------------------------------------------------
    #  Sub-tabs
    # ------------------------------------------------------------------
    pg_tab1, pg_tab2, pg_tab3 = st.tabs([
        "📄 Model View", "✏️ Live Editor", "📊 Sensitivity Sweep"
    ])

    # ==================================================================
    #  Sub-tab 1: Model View (read-only rendered LaTeX)
    # ==================================================================
    with pg_tab1:
        st.subheader("Current Mathematical Model")
        _sense_label = _pg_orig.get('sense', 'minimize').capitalize()
        st.markdown(f"**Objective ({_sense_label}):**")
        st.latex(_pg_orig.get('objective_latex', ''))

        if _pg_orig.get('constraints'):
            st.markdown("**Subject to:**")
            for _ci, _con in enumerate(_pg_orig['constraints']):
                st.latex(_con.get('latex', ''))

        if _pg_orig.get('variables'):
            st.markdown("**Decision Variables:**")
            _var_rows = []
            for _v in _pg_orig['variables']:
                _lb = _v.get('lb', 0) if _v.get('lb') is not None else '-∞'
                _ub = _v.get('ub') if _v.get('ub') is not None else '+∞'
                _var_rows.append({
                    'Variable': _v['name'],
                    'LaTeX': f"${_v.get('sympy_name', _v['name'])}$",
                    'Type': _v.get('cat', 'Continuous'),
                    'Lower Bound': _lb,
                    'Upper Bound': _ub,
                })
            st.dataframe(pd.DataFrame(_var_rows), width="stretch", hide_index=True)

        # Show current solution values if available
        _orig_sol = st.session_state.get('playground_original_solution')
        if _orig_sol and _orig_sol.get('variable_values'):
            st.markdown("**Current Solution:**")
            _sol_rows = []
            for _vname, _vval in _orig_sol['variable_values'].items():
                _sol_rows.append({'Variable': _vname, 'Value': _vval})
            st.dataframe(pd.DataFrame(_sol_rows), width="stretch", hide_index=True)
            _obj_val = _orig_sol.get('objective_value')
            if _obj_val is not None:
                st.metric("Objective Value", f"{_obj_val:,.4f}")

    # ==================================================================
    #  Sub-tab 2: Live Editor
    # ==================================================================
    with pg_tab2:
        st.subheader("Edit Model")
        st.caption(
            "Modify the objective, constraints, and variables below. "
            "Use the math toolbar to insert symbols, or type LaTeX directly."
        )

        # -- Objective --------------------------------------------------
        st.markdown("---")
        st.markdown("#### Objective Function")
        _sense_options = ['minimize', 'maximize']
        _sense_idx = _sense_options.index(_pg_edit.get('sense', 'minimize'))
        _new_sense = st.selectbox(
            "Direction", _sense_options, index=_sense_idx, key="pg_sense_select"
        )
        _pg_edit['sense'] = _new_sense

        _obj_input = mathlive_editor(
            label="Objective Expression",
            initial_latex=_pg_edit.get('objective_latex', ''),
            height=180,
            key="pg_obj",
            toolbar_mode='or_full',
        )
        if _obj_input:
            _pg_edit['objective_latex'] = _obj_input

        # -- Constraints ------------------------------------------------
        st.markdown("---")
        st.markdown("#### Constraints")

        _constraints_to_remove = []
        for _ci, _con in enumerate(_pg_edit.get('constraints', [])):
            with st.expander(f"Constraint: {_con.get('name', f'C{_ci+1}')}", expanded=False):
                _cname = st.text_input(
                    "Name", value=_con.get('name', f'constraint_{_ci}'),
                    key=f"pg_con_name_{_ci}",
                )
                _con['name'] = _cname

                _con_edited = mathlive_editor(
                    label=f"Constraint {_ci + 1}",
                    initial_latex=_con.get('latex', ''),
                    height=160,
                    key=f"pg_con_{_ci}",
                    toolbar_mode='minimal',
                )
                if _con_edited and _con_edited != _con.get('latex', ''):
                    _con['latex'] = _con_edited
                    parsed_con = parse_constraint_latex(_con_edited)
                    _con['sense'] = parsed_con['sense']
                    if parsed_con.get('rhs_latex'):
                        from src.utils.model_editor import _parse_rhs_value
                        _parsed_rhs = _parse_rhs_value(parsed_con['rhs_latex'])
                        if _parsed_rhs is not None:
                            _con['rhs'] = _parsed_rhs
                            st.session_state[f"pg_con_rhs_{_ci}"] = float(_parsed_rhs)

                with st.form(key=f"pg_con_form_{_ci}", clear_on_submit=False):
                    _form_rhs = st.number_input(
                        "Constraint limit (RHS)",
                        value=float(_con.get('rhs', 0)),
                        key=f"pg_con_rhs_{_ci}", step=1.0,
                    )
                    _rhs_submitted = st.form_submit_button(
                        "✅ Apply RHS Change", width="stretch",
                    )
                if _rhs_submitted:
                    _con['rhs'] = _form_rhs

                if st.button("🗑️ Remove", key=f"pg_con_del_{_ci}"):
                    _constraints_to_remove.append(_ci)

        for _idx in reversed(_constraints_to_remove):
            _pg_edit['constraints'].pop(_idx)
        if _constraints_to_remove:
            st.rerun()

        with st.expander("➕ Add New Constraint"):
            _new_con_name = st.text_input(
                "Constraint name", value=f"new_constraint_{len(_pg_edit.get('constraints', []))+1}",
                key="pg_new_con_name",
            )
            _new_con_latex = mathlive_editor(
                label="New constraint expression",
                initial_latex=r"x_{1} + x_{2} \leq 100",
                height=140,
                key="pg_new_con_editor",
                toolbar_mode='minimal',
            )
            _new_con_rhs = st.number_input(
                "Constraint limit (RHS)", value=100.0, key="pg_new_con_rhs", step=1.0,
            )
            if st.button("Add Constraint", key="pg_add_con"):
                _latex = _new_con_latex or r"x_{1} \leq 100"
                _pg_edit.setdefault('constraints', []).append({
                    'name': _new_con_name,
                    'latex': _latex,
                    'sense': parse_constraint_latex(_latex).get('sense', '<='),
                    'rhs': _new_con_rhs,
                })
                st.rerun()

        # -- Variables ---------------------------------------------------
        st.markdown("---")
        st.markdown("#### Variables")

        _vars_list = _pg_edit.get('variables', [])
        _vars_to_remove = []

        with st.form("pg_variables_form", clear_on_submit=False):
            _form_var_edits = {}
            for _vi, _vinfo in enumerate(_vars_list):
                _vcols = st.columns([2, 2, 1, 1, 1])
                with _vcols[0]:
                    st.text_input(
                        "Name", value=_vinfo['name'],
                        key=f"pg_var_name_{_vi}", disabled=True,
                    )
                with _vcols[1]:
                    st.text(f"${_vinfo.get('sympy_name', '')}$")
                with _vcols[2]:
                    _cat_options = ['Continuous', 'Integer', 'Binary']
                    _cat_idx = _cat_options.index(_vinfo.get('cat', 'Continuous')) if _vinfo.get('cat') in _cat_options else 0
                    _new_cat = st.selectbox(
                        "Type", _cat_options, index=_cat_idx, key=f"pg_var_cat_{_vi}",
                    )
                with _vcols[3]:
                    _new_lb = st.number_input(
                        "LB", value=float(_vinfo.get('lb', 0) or 0),
                        key=f"pg_var_lb_{_vi}", step=1.0,
                    )
                with _vcols[4]:
                    _ub_val = _vinfo.get('ub')
                    _is_unbounded = _ub_val is None or _ub_val >= 999999
                    _new_ub = st.number_input(
                        "UB", value=0.0 if _is_unbounded else float(_ub_val),
                        key=f"pg_var_ub_{_vi}", step=1.0,
                    )
                    if _is_unbounded:
                        st.caption("0 = no upper bound (∞)")
                _form_var_edits[_vi] = {
                    'cat': _new_cat, 'lb': _new_lb, 'ub_raw': _new_ub,
                }

            _vars_submitted = st.form_submit_button(
                "✅ Apply Variable Changes", width="stretch",
            )

        if _vars_submitted:
            for _vi, _edits in _form_var_edits.items():
                if _vi < len(_vars_list):
                    _vars_list[_vi]['cat'] = _edits['cat']
                    _vars_list[_vi]['lb'] = _edits['lb']
                    _vars_list[_vi]['ub'] = None if _edits['ub_raw'] == 0 else _edits['ub_raw']

        if _vars_list:
            st.caption("Remove a variable:")
            _n_cols = max(1, min(len(_vars_list), 6))
            _del_cols = st.columns(_n_cols)
            for _vi, _vinfo in enumerate(_vars_list):
                with _del_cols[_vi % _n_cols]:
                    if st.button(f"🗑️ {_vinfo['name']}", key=f"pg_var_del_{_vi}"):
                        _vars_to_remove.append(_vi)

        for _idx in reversed(_vars_to_remove):
            _pg_edit['variables'].pop(_idx)
        if _vars_to_remove:
            st.rerun()

        with st.expander("➕ Add New Variable"):
            _nv_cols = st.columns(4)
            with _nv_cols[0]:
                _nv_name = st.text_input("Name", value="x_new", key="pg_new_var_name")
            with _nv_cols[1]:
                _nv_cat = st.selectbox("Type", ['Continuous', 'Integer', 'Binary'], key="pg_new_var_cat")
            with _nv_cols[2]:
                _nv_lb = st.number_input("Lower bound", value=0.0, key="pg_new_var_lb", step=1.0)
            with _nv_cols[3]:
                _nv_ub = st.number_input("Upper bound", value=1e6, key="pg_new_var_ub", step=1.0)
            if st.button("Add Variable", key="pg_add_var"):
                from src.utils.model_editor import _pulp_var_to_sympy_name
                _pg_edit.setdefault('variables', []).append({
                    'name': _nv_name,
                    'sympy_name': _pulp_var_to_sympy_name(_nv_name),
                    'lb': _nv_lb,
                    'ub': _nv_ub if _nv_ub < 1e6 else None,
                    'cat': _nv_cat,
                })
                st.rerun()

        # -- Action buttons ---------------------------------------------
        st.markdown("---")
        _act_cols = st.columns(3)
        with _act_cols[0]:
            _solve_edited = st.button(
                "🚀 Solve Edited Model", type="primary", key="pg_solve",
                width="stretch",
            )
        with _act_cols[1]:
            if st.button("👁️ Preview as LaTeX", key="pg_preview", width="stretch"):
                st.markdown("**Objective:**")
                st.latex(_pg_edit.get('objective_latex', ''))
                for _con in _pg_edit.get('constraints', []):
                    st.latex(_con.get('latex', ''))
        with _act_cols[2]:
            if st.button("🔄 Reset to Original", key="pg_reset", width="stretch"):
                st.session_state.playground_editor_state = deepcopy(
                    st.session_state.playground_original_state
                )
                if 'playground_edited_solution' in st.session_state:
                    del st.session_state['playground_edited_solution']
                st.rerun()

        # -- Solve edited model -----------------------------------------
        if _solve_edited:
            with st.spinner("Building and solving edited model..."):
                try:
                    _edited_model, _build_warnings = editor_state_to_pulp(_pg_edit)

                    if _build_warnings:
                        for _w in _build_warnings:
                            st.warning(f"⚠️ {_w}")

                    _pg_problem_data = st.session_state.get('problem_data', {})
                    _solver_type = st.session_state.get('solver_key', 'auto')
                    _max_time = st.session_state.get('max_time', 120)

                    _pg_solver = SolverInterface(
                        solver_type=_solver_type,
                        problem_data=_pg_problem_data,
                    )
                    _edited_solution = _pg_solver.solve(_edited_model, max_time=_max_time)
                    st.session_state.playground_edited_solution = _edited_solution

                except Exception as e:
                    st.error(f"Failed to solve edited model: {e}")
                    _edited_solution = None

        # -- Comparison table -------------------------------------------
        _edited_sol = st.session_state.get('playground_edited_solution')
        if _edited_sol:
            st.markdown("---")
            st.subheader("📊 Comparison: Original vs Edited")

            _changes = diff_states(_pg_orig, _pg_edit)

            _orig_sol = st.session_state.get('playground_original_solution', {})
            _orig_obj = _orig_sol.get('objective_value') if _orig_sol else None
            _edit_obj = _edited_sol.get('objective_value')

            _cmp_cols = st.columns(3)
            with _cmp_cols[0]:
                _orig_display = f"{_orig_obj:,.4f}" if _orig_obj is not None else "N/A"
                st.metric("Original Objective", _orig_display)
            with _cmp_cols[1]:
                _edit_display = f"{_edit_obj:,.4f}" if _edit_obj is not None else "N/A"
                st.metric("Edited Objective", _edit_display)
            with _cmp_cols[2]:
                if _orig_obj is not None and _edit_obj is not None:
                    _delta = _edit_obj - _orig_obj
                    _pct = (_delta / abs(_orig_obj) * 100) if _orig_obj != 0 else 0
                    st.metric("Delta", f"{_delta:,.4f}", f"{_pct:+.1f}%")
                else:
                    st.metric("Delta", "N/A")

            if _changes and _changes != ["No changes detected"]:
                st.markdown("**Changes Made:**")
                for _ch in _changes:
                    st.markdown(f"- {_ch}")

            # Variable comparison
            _orig_vals = _orig_sol.get('variable_values', {}) if _orig_sol else {}
            _edit_vals = _edited_sol.get('variable_values', {})
            if _orig_vals or _edit_vals:
                _all_vars = sorted(set(list(_orig_vals.keys()) + list(_edit_vals.keys())))
                _cmp_rows = []
                for _vn in _all_vars:
                    _ov = _orig_vals.get(_vn)
                    _ev = _edit_vals.get(_vn)
                    _cmp_rows.append({
                        'Variable': _vn,
                        'Original': f"{_ov:.4f}" if _ov is not None else "—",
                        'Edited': f"{_ev:.4f}" if _ev is not None else "—",
                        'Delta': f"{_ev - _ov:+.4f}" if (_ov is not None and _ev is not None) else "—",
                    })
                st.dataframe(
                    pd.DataFrame(_cmp_rows),
                    width="stretch",
                    hide_index=True,
                )

    # ==================================================================
    #  Sub-tab 3: Sensitivity Sweep
    # ==================================================================
    with pg_tab3:
        st.subheader("Constraint RHS Sensitivity Sweep")
        st.caption(
            "Pick a constraint, sweep its RHS across a range, and observe "
            "how the optimal objective changes. The slope of the line "
            "approximates the **average marginal value** (for integer models "
            "this is an approximation, not a true shadow price)."
        )

        _cons = _pg_edit.get('constraints', [])
        if not _cons:
            st.warning("No constraints in the current model.")
            st.stop()

        _con_names = [c.get('name', f'C{i+1}') for i, c in enumerate(_cons)]
        
        # ISSUE 5 FIX: Use session state to update defaults when constraint changes
        def _on_constraint_change():
            _sel_idx = _con_names.index(st.session_state.pg_sweep_con)
            _current_rhs = _cons[_sel_idx].get('rhs', 0)
            st.session_state.pg_sweep_min_default = float(max(0, _current_rhs * 0.2))
            st.session_state.pg_sweep_max_default = float(_current_rhs * 2.5) if _current_rhs else 100.0
        
        _selected_con = st.selectbox(
            "Select constraint", _con_names, key="pg_sweep_con", 
            on_change=_on_constraint_change
        )
        _sel_idx = _con_names.index(_selected_con)
        _current_rhs = _cons[_sel_idx].get('rhs', 0)
        
        # Initialize defaults if not set
        if 'pg_sweep_min_default' not in st.session_state:
            st.session_state.pg_sweep_min_default = float(max(0, _current_rhs * 0.2))
            st.session_state.pg_sweep_max_default = float(_current_rhs * 2.5) if _current_rhs else 100.0

        with st.form("pg_sweep_form", clear_on_submit=False):
            _sw_cols = st.columns(3)
            with _sw_cols[0]:
                _rhs_min = st.number_input(
                    "RHS Min", value=st.session_state.pg_sweep_min_default,
                    key="pg_sweep_min", step=1.0,
                )
            with _sw_cols[1]:
                _rhs_max = st.number_input(
                    "RHS Max", value=st.session_state.pg_sweep_max_default,
                    key="pg_sweep_max", step=1.0,
                )
            with _sw_cols[2]:
                _n_points = st.number_input(
                    "Number of points", value=10, min_value=3, max_value=50,
                    key="pg_sweep_n", step=1,
                )
            _sweep_submitted = st.form_submit_button(
                "🔍 Run Sweep", type="primary", width="stretch",
            )

        if _sweep_submitted:
            import numpy as np

            _sweep_cache_key = f"sweep_{_selected_con}_{_rhs_min}_{_rhs_max}_{_n_points}"
            _sweep_cache = st.session_state.get('_sweep_cache', {})

            if _sweep_cache_key in _sweep_cache:
                _rhs_vals, _obj_vals = _sweep_cache[_sweep_cache_key]
            else:
                _rhs_vals = np.linspace(_rhs_min, _rhs_max, int(_n_points)).tolist()
                _obj_vals = []
                _progress = st.progress(0, text="Sweeping...")

                for _pi, _rv in enumerate(_rhs_vals):
                    _sweep_state = deepcopy(_pg_edit)
                    _sweep_state['constraints'][_sel_idx]['rhs'] = _rv

                    try:
                        _sweep_model, _ = editor_state_to_pulp(_sweep_state)
                        _pg_problem_data = st.session_state.get('problem_data', {})
                        _sw_solver = SolverInterface(
                            solver_type=st.session_state.get('solver_key', 'auto'),
                            problem_data=_pg_problem_data,
                        )
                        _sw_sol = _sw_solver.solve(_sweep_model, max_time=30)
                        _ov = _sw_sol.get('objective_value')
                        _obj_vals.append(_ov if _ov is not None else float('nan'))
                    except Exception:
                        _obj_vals.append(float('nan'))

                    _progress.progress(
                        (_pi + 1) / len(_rhs_vals),
                        text=f"Solving {_pi+1}/{len(_rhs_vals)}...",
                    )
                _progress.empty()

                # Cache (limit to 150 entries)
                if len(_sweep_cache) >= 150:
                    _oldest = next(iter(_sweep_cache))
                    del _sweep_cache[_oldest]
                _sweep_cache[_sweep_cache_key] = (_rhs_vals, _obj_vals)
                st.session_state['_sweep_cache'] = _sweep_cache

            # Plot
            _sweep_df = pd.DataFrame({
                'RHS Value': _rhs_vals,
                'Objective': _obj_vals,
            })
            _fig = px.line(
                _sweep_df, x='RHS Value', y='Objective',
                title=f"Sensitivity: {_selected_con} RHS vs Objective",
                markers=True,
            )
            _fig.add_vline(
                x=_current_rhs, line_dash="dash", line_color="red",
                annotation_text=f"Current: {_current_rhs}",
            )
            st.plotly_chart(_fig, width="stretch")

            # Shadow price approximation
            import numpy as np
            _valid = [
                (r, o) for r, o in zip(_rhs_vals, _obj_vals)
                if o is not None and not (isinstance(o, float) and np.isnan(o))
            ]
            if len(_valid) >= 2:
                _r_arr = [v[0] for v in _valid]
                _o_arr = [v[1] for v in _valid]
                _slope = (
                    (_o_arr[-1] - _o_arr[0]) / (_r_arr[-1] - _r_arr[0])
                    if (_r_arr[-1] - _r_arr[0]) != 0 else 0
                )
                st.metric(
                    "Average Marginal Value",
                    f"{_slope:,.4f}",
                    help="Average change in objective per unit change in constraint RHS. Note: For integer/binary models, the actual marginal value is step-wise, not continuous.",
                )
            else:
                st.info("Not enough feasible points to compute a shadow price.")
