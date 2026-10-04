"""
MathLive Streamlit Component (bi-directional)

Embeds MathLive (https://cortexjs.io/mathlive/) as an interactive
math editor inside Streamlit using ``declare_component``.

MathLive loads from CDN. Edits in the visual math editor are sent
directly back to Python via Streamlit's component messaging —
no text_input bridge needed.
"""

import streamlit.components.v1 as components
from pathlib import Path

_component_func = components.declare_component(
    "mathlive_editor",
    path=str(Path(__file__).parent / "mathlive_build"),
)


_TOOLBAR_OR_FULL = r"""<div class="or-toolbar" id="toolbar">
    <div class="or-toolbar-section">
      <button class="tb-btn" data-cmd='["insert","\\sum_{#0}^{#0}#0"]' data-tip="Summation">&sum;</button>
      <button class="tb-btn" data-cmd='["insert","\\prod_{#0}^{#0}#0"]' data-tip="Product">&prod;</button>
      <button class="tb-btn" data-cmd='["insert","\\int_{#0}^{#0}#0"]' data-tip="Integral">&int;</button>
    </div>
    <div class="or-toolbar-divider"></div>
    <div class="or-toolbar-section">
      <button class="tb-btn" data-cmd='["insert","\\leq"]' data-tip="Less or equal">&le;</button>
      <button class="tb-btn" data-cmd='["insert","\\geq"]' data-tip="Greater or equal">&ge;</button>
      <button class="tb-btn" data-cmd='["insert","="]' data-tip="Equal">=</button>
      <button class="tb-btn" data-cmd='["insert","\\neq"]' data-tip="Not equal">&ne;</button>
    </div>
    <div class="or-toolbar-divider"></div>
    <div class="or-toolbar-section">
      <button class="tb-btn" data-cmd='["insert","\\in"]' data-tip="Element of">&isin;</button>
      <button class="tb-btn" data-cmd='["insert","\\mathbb{Z}"]' data-tip="Integers">&#8484;</button>
      <button class="tb-btn" data-cmd='["insert","\\mathbb{R}"]' data-tip="Reals">&#8477;</button>
      <button class="tb-btn" data-cmd='["insert","\\{0,1\\}"]' data-tip="Binary set">{0,1}</button>
      <button class="tb-btn" data-cmd='["insert","\\infty"]' data-tip="Infinity">&infin;</button>
    </div>
    <div class="or-toolbar-divider"></div>
    <div class="or-toolbar-section">
      <button class="tb-btn" data-cmd='["insert","\\frac{#0}{#0}"]' data-tip="Fraction">&frac12;</button>
      <button class="tb-btn" data-cmd='["insert","#0^{#0}"]' data-tip="Superscript">x&#x207F;</button>
      <button class="tb-btn" data-cmd='["insert","#0_{#0}"]' data-tip="Subscript">x&#x1D62;</button>
      <button class="tb-btn" data-cmd='["insert","\\sqrt{#0}"]' data-tip="Square root">&radic;</button>
    </div>
    <div class="or-toolbar-divider"></div>
    <div class="or-toolbar-section">
      <button class="tb-btn" data-cmd='["insert","\\alpha"]' data-tip="alpha">&alpha;</button>
      <button class="tb-btn" data-cmd='["insert","\\beta"]' data-tip="beta">&beta;</button>
      <button class="tb-btn" data-cmd='["insert","\\lambda"]' data-tip="lambda">&lambda;</button>
      <button class="tb-btn" data-cmd='["insert","\\mu"]' data-tip="mu">&mu;</button>
      <button class="tb-btn" data-cmd='["insert","\\delta"]' data-tip="delta">&delta;</button>
      <button class="tb-btn" data-cmd='["insert","\\epsilon"]' data-tip="epsilon">&epsilon;</button>
    </div>
    <div class="or-toolbar-divider"></div>
    <div class="or-toolbar-section">
      <button class="tb-btn" data-cmd='["insert","\\sum_{i=1}^{n}c_{i}x_{i}"]' data-tip="LP objective template" style="font-size:11px">LP obj</button>
      <button class="tb-btn" data-cmd='["insert","\\sum_{j=1}^{n}x_{ij}\\leq s_{i}"]' data-tip="Supply constraint" style="font-size:11px">Supply</button>
      <button class="tb-btn" data-cmd='["insert","\\sum_{i=1}^{m}x_{ij}=d_{j}"]' data-tip="Demand constraint" style="font-size:11px">Demand</button>
      <button class="tb-btn" data-cmd='["insert","\\sum_{i=1}^{n}w_{i}x_{i}\\leq W"]' data-tip="Knapsack capacity" style="font-size:11px">Knapsack</button>
    </div>
  </div>"""

_TOOLBAR_MINIMAL = r"""<div class="or-toolbar" id="toolbar">
    <div class="or-toolbar-section">
      <button class="tb-btn" data-cmd='["insert","\\sum_{#0}^{#0}#0"]'>&sum;</button>
      <button class="tb-btn" data-cmd='["insert","\\leq"]'>&le;</button>
      <button class="tb-btn" data-cmd='["insert","\\geq"]'>&ge;</button>
      <button class="tb-btn" data-cmd='["insert","\\frac{#0}{#0}"]'>&frac12;</button>
      <button class="tb-btn" data-cmd='["insert","#0^{#0}"]'>x&#x207F;</button>
      <button class="tb-btn" data-cmd='["insert","#0_{#0}"]'>x&#x1D62;</button>
    </div>
  </div>"""


def mathlive_editor(
    label: str,
    initial_latex: str = "",
    height: int = 120,
    key: str = "mathlive",
    read_only: bool = False,
    toolbar_mode: str = "or_full",
) -> str | None:
    """Render a MathLive editor and return the current LaTeX string.

    Uses ``declare_component`` for true bi-directional communication.
    Edits inside the visual math field are sent directly to Python.
    Returns the current LaTeX, or *initial_latex* before any interaction.
    """
    if toolbar_mode == "or_full":
        toolbar_html = _TOOLBAR_OR_FULL
    elif toolbar_mode == "minimal":
        toolbar_html = _TOOLBAR_MINIMAL
    else:
        toolbar_html = ""

    import streamlit as st

    current = st.session_state.get(key)
    effective_latex = current if current else initial_latex

    result = _component_func(
        label=label,
        initial_latex=effective_latex,
        toolbar_html=toolbar_html,
        read_only=read_only,
        height=height,
        key=key,
        default=initial_latex,
    )
    return result
