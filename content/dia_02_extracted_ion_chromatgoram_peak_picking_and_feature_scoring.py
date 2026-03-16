from __future__ import annotations

import inspect

import streamlit as st
from src.common.common import page_setup

page_setup()


st.title("Extracted Ion Chromatogram (XIC) Peak Picking and Feature Scoring")
st.markdown(
    """
This page will walk you through the typical concepts for performing peak picking and feature scoring on extracted ion chromatograms (XICs) in DIA data.
We will use the same small sample dataset (as the previous section) from the [DIA PASEF Evosep dataset (PXD017703)](https://www.ebi.ac.uk/pride/archive/projects/PXD017703) published by Meier et al., 2020. This dataset contains DIA data acquired on a Bruker timsTOF Pro instrument using the Evosep One LC system. The sample file we will use is a small subset of the full experiment run (*20200505_Evosep_200SPD_SG06-16_MLHeLa_200ng_py8_S3-A1_1_2737*), containing spectra in the MS1 precursor m/z range 660-700 and RT range 130-170s. 
"""
)

st.markdown("---")