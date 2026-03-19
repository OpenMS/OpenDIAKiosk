from __future__ import annotations


import streamlit as st

from src.common.common import page_setup

page_setup()

# -----------------------------------------------------------------------------
# Page content

st.title("Statistical Validation of DIA Features")
st.markdown(
    """
This page demonstrates **semi-supervised statistical validation** of DIA feature scores.

The workflow mirrors a typical workflow in PyProphet:
1. Rank all candidate peak groups by **main score** (`main_var_xcorr_shape`)
2. Select **confident target training examples** (top-1 per group passing an FDR threshold) + all decoys
3. Train a discriminant model on this enriched training set
4. Re-score all features, tighten the FDR, repeat for several iterations
5. Normalise the final score by the decoy distribution → **d-score**
6. Compute **empirical p-values**, **π₀** (bootstrap), **q-values** and **PEP**
"""
)
