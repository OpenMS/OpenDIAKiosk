from __future__ import annotations

import inspect

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pyopenms as poms
import pyopenms_viz  # noqa: F401  # registers plotting backends for pandas

import streamlit as st
from src.common.common import page_setup
from utils.dia_tutorial import (
    reduce_spectra,
    annotate_filtered_spectra,
    apply_sgolay,
    msexperiment_to_dataframe,
)
from utils.dia_peak_picking import (
    smooth_chromatogram,
    create_concensus_chromatogram,
    perform_xic_peak_picking,
    merge_transition_peak_boundaries_to_consensus,
)

page_setup()


st.title("Extracted Ion Chromatogram (XIC) Peak Picking and Feature Scoring")
st.markdown(
    """
This page will walk you through the typical concepts for performing peak picking and feature scoring on extracted ion chromatograms (XICs) in DIA data.
We will use the same small sample dataset (as the previous section) from the [DIA PASEF Evosep dataset (PXD017703)](https://www.ebi.ac.uk/pride/archive/projects/PXD017703) published by Meier et al., 2020. This dataset contains DIA data acquired on a Bruker timsTOF Pro instrument using the Evosep One LC system. The sample file we will use is a small subset of the full experiment run (*20200505_Evosep_200SPD_SG06-16_MLHeLa_200ng_py8_S3-A1_1_2737*), containing spectra in the MS1 precursor m/z range 660-700 and RT range 130-170s. 
"""
)

st.markdown("---")
st.subheader("Load mzML Data")
mz_file = "example-data/mzML/20200505_Evosep_200SPD_SG06-16_filtered_ms1_mz_660_700_rt_130_170_with_ms2.mzML.gz"

peptide = "NTGIIC(UniMod:4)TIGPASR"
precursor_mz = 680.3561
precursor_charge = 2
product_mzs = [386.2034, 487.2623, 600.3464, 701.3941, 861.4247, 974.5088]
product_charges = [1, 1, 1, 1, 1, 1]
product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
prec_mz_tol = 15
prod_mz_tol = 20
target_im = 0.96756938061
im_tol = 0.08

# Session state variables to store the loaded data and prevent re-loading on every button click
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "exp_df_targeted" not in st.session_state:
    st.session_state.exp_df_targeted = None
if "sgolay_frame_length" not in st.session_state:
    st.session_state.sgolay_frame_length = 9
if "sgolay_polynomial_order" not in st.session_state:
    st.session_state.sgolay_polynomial_order = 3
if "gaussian_peak_width" not in st.session_state:
    st.session_state.gaussian_peak_width = 50.0
if "picked_peaks_df" not in st.session_state:
    st.session_state.picked_peaks_df = None

load_clicked = st.button(
    "Load DIA Data and perform targeted extraction", type="primary"
)

if load_clicked or st.session_state.data_loaded:
    with st.spinner("Loading and extracting targeted XICs..."):
        progress = st.progress(0)
        if not st.session_state.data_loaded:
            exp = poms.MSExperiment()
            poms.MzMLFile().load(mz_file, exp)
            progress.progress(20)
            exp_df = exp.to_df(long_format=True)
            progress.progress(30)

            filtered_exp = reduce_spectra(
                exp,
                float(precursor_mz),
                product_mzs,
                float(prec_mz_tol),
                float(prod_mz_tol),
                float(target_im),
                float(im_tol),
            )
            progress.progress(55)

            filtered_exp.updateRanges()
            progress.progress(65)
            exp_df_targeted = filtered_exp.to_df(long_format=True)
            progress.progress(75)

            exp_df_targeted = annotate_filtered_spectra(
                filtered_df=exp_df_targeted,
                precursor_mz=float(precursor_mz),
                precursor_charge=int(precursor_charge),
                product_mzs=product_mzs,
                product_charges=product_charges,
                product_annotations=product_annotations,
                prec_mz_tol=float(prec_mz_tol),
                prod_mz_tol=float(prod_mz_tol),
            )
            st.session_state.data_loaded = True
            st.session_state.exp_df_targeted = exp_df_targeted
            progress.progress(90)
        else:
            exp_df_targeted = st.session_state.exp_df_targeted

        group_cols = ["ms_level", "annotation", "rt"]
        integrate_col = "intensity"
        smoothed_chrom_fig = (
            exp_df_targeted.apply(
                lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
            )
            .groupby(group_cols)[integrate_col]
            .sum()
            .reset_index()
            .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
            .apply(apply_sgolay, window_length=9, polyorder=3)
            .reset_index(drop=True)
            .plot(
                kind="chromatogram",
                x="rt",
                y="smoothed_int",
                by="annotation",
                title=f"Smoothed Chromatogram - {peptide} / {precursor_charge}+",
                aggregate_duplicates=False,
                legend_config=dict(title="Transition"),
                backend="ms_plotly",
                show_plot=False,
            )
        )

        progress.progress(100)
        st.plotly_chart(smoothed_chrom_fig, use_container_width=True)

    st.markdown("""
                From the previous section, we learned how to perform targeted data extraction to filter the spectra in our DIA dataset to only those relevant to our peptide of interest (NTGIIC(UniMod:4)TIGPASR) and its corresponding transitions, and we saw how we can visualize the extracted ion chromatograms (XICs) for each transition. However, we have not yet performed any peak picking or feature scoring on these XICs, which are critical steps for determining the quality of our detected features and for downstream quantification and statistical analysis.
                
                Visually, it's pretty easy to tell that this peptide has a good co-elution of its transitions, with a clear peak around 152s. However, when processing thousands of peptides and features in a typical DIA dataset, we need to be able to perform automated peak picking and feature scoring to determine which features are likely to be true positives and which are likely to be false positives. This is where peak picking algorithms and feature scoring metrics come into play.
                """)

    with st.expander("Code:"):
        st.code(
            """exp = poms.MSExperiment()
    poms.MzMLFile().load(mz_file, exp)
    progress.progress(20)
    exp_df = exp.to_df(long_format=True)
    progress.progress(30)

    peptide = "NTGIIC(UniMod:4)TIGPASR"
    precursor_mz = 680.3561
    precursor_charge = 2
    product_mzs = [386.2034, 487.2623, 600.3464, 701.3941, 861.4247, 974.5088]
    product_charges = [1, 1, 1, 1, 1, 1]
    product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
    prec_mz_tol = 15
    prod_mz_tol = 20
    target_im = 0.96756938061
    im_tol = 0.08

    filtered_exp = reduce_spectra(
        exp,
        float(precursor_mz),
        product_mzs,
        float(prec_mz_tol),
        float(prod_mz_tol),
        float(target_im),
        float(im_tol),
    )
    progress.progress(55)

    filtered_exp.updateRanges()
    progress.progress(65)
    exp_df_targeted = filtered_exp.to_df(long_format=True)
    progress.progress(75)

    exp_df_targeted = annotate_filtered_spectra(
        filtered_df=exp_df_targeted,
        precursor_mz=float(precursor_mz),
        precursor_charge=int(precursor_charge),
        product_mzs=product_mzs,
        product_charges=product_charges,
        product_annotations=product_annotations,
        prec_mz_tol=float(prec_mz_tol),
        prod_mz_tol=float(prod_mz_tol),
    )
    progress.progress(90)

    group_cols = ["ms_level", "annotation", "rt"]
    integrate_col = "intensity"
    smoothed_chrom_fig = (
        exp_df_targeted.apply(
            lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
        )
        .groupby(group_cols)[integrate_col]
        .sum()
        .reset_index()
        .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
        .apply(apply_sgolay, window_length=9, polyorder=3)
        .reset_index(drop=True)
        .plot(
            kind="chromatogram",
            x="rt",
            y="smoothed_int",
            by="annotation",
            title=f"Smoothed Chromatogram - {peptide} / {precursor_charge}+",
            aggregate_duplicates=False,
            legend_config=dict(title="Transition"),
            backend="ms_plotly",
            show_plot=False,
        )
    )
    """,
            language="python",
        )


st.markdown("---")
st.subheader("Peak Picking")

st.markdown(
    """
    In traditional DIA data processing pipelines, peak picking is often performed on the extracted ion chromatograms (XICs) for each transition to identify potential chromatographic peaks present in each transition. This results in a set of identified peak boundaries for each transition, which are then merged to generate concensus peak boundaries for each precursor feature. 
    """
)

# Plot each XIC trace on a single row in a subplot, instead of having all transitions plotted on the same axes, to better visualize the individual traces
# iterate over the df by the annotation column, and plot each transition in a separate subplot row, with the same x-axis (rt) and y-axis (intensity),
if (
    st.session_state.exp_df_targeted is not None
    and not st.session_state.exp_df_targeted.empty
):
    # buttom to perform peak picking on the smoothed dataframe
    peak_picking_clicked = st.button(
        "Perform Peak Picking on Smoothed XICs", type="primary"
    )

    picker = poms.PeakPickerChromatogram()
    picker_params = picker.getDefaults()
    # st.write(picker_params.to_dict())

    st.markdown("#### Peak Picking Parameters")
    smoothing_method = st.selectbox(
        "Smoothing method",
        options=["Savitzky-Golay", "Gaussian", "Raw"],
        index=0,
        key="smoothing_method",
        help="Select the smoothing method to apply to the XICs before peak picking. 'Raw' will perform peak picking on the unsmoothed data.",
    )

    if smoothing_method == "Savitzky-Golay":
        col1, col2 = st.columns(2)
        with col1:
            st.slider(
                "Savitzky-Golay Frame Length",
                min_value=3,
                max_value=21,
                step=2,
                value=9,
                key="sgolay_frame_length",
                help="The window length for the Savitzky-Golay filter. Must be an odd integer.",
            )
        with col2:
            st.slider(
                "Savitzky-Golay Polynomial Order",
                min_value=1,
                max_value=21,
                step=1,
                value=3,
                key="sgolay_polynomial_order",
                help="The polynomial order for the Savitzky-Golay filter. Must be less than the frame length.",
            )
    elif smoothing_method == "Gaussian":
        st.slider(
            "Gaussian Peak Width (s)",
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            value=50.0,
            key="gaussian_peak_width",
            help="The width of the Gaussian kernel for Gaussian smoothing.",
        )

    if smoothing_method in ["Savitzky-Golay", "Gaussian"]:
        picker_params.setValue("method", "corrected")
        if smoothing_method == "Savitzky-Golay":
            picker_params.setValue(
                "sgolay_frame_length", st.session_state.sgolay_frame_length
            )
            picker_params.setValue(
                "sgolay_polynomial_order", st.session_state.sgolay_polynomial_order
            )
        elif smoothing_method == "Gaussian":
            picker_params.setValue("use_gauss", "true")
            picker_params.setValue("gauss_width", st.session_state.gaussian_peak_width)
    else:
        picker_params.setValue("method", "legacy")

    picker.setParameters(picker_params)

    if peak_picking_clicked or st.session_state.picked_peaks_df is not None:
        with st.spinner("Performing peak picking and preparing plots..."):
            progress_pp = st.progress(0)

            try:
                summed_xic_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(group_cols)[integrate_col]
                    .sum()
                    .reset_index()
                    .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                    .apply(apply_sgolay, window_length=9, polyorder=3)
                    .reset_index(drop=True)
                )
                st.session_state.picked_peaks_df = perform_xic_peak_picking(
                    summed_xic_df, intensity_col="intensity", picker=picker
                )
                st.dataframe(st.session_state.picked_peaks_df)
            except Exception as e:
                st.error(f"Error performing peak picking: {e}")
                st.stop()
            progress_pp.progress(50)

            smoothed_chromatogram_df = smooth_chromatogram(
                summed_xic_df,
                smoothing_method=smoothing_method,
                sgolay_window=st.session_state.sgolay_frame_length,
                sgolay_polyorder=st.session_state.sgolay_polynomial_order,
                gauss_width=st.session_state.gaussian_peak_width,
            )

            # per-transition subplot: one row per annotation, shared x-axis
            annotations = list(smoothed_chromatogram_df["annotation"].unique())
            fig_sub = make_subplots(
                rows=len(annotations), cols=1, shared_xaxes=True, vertical_spacing=0.02
            )
            for i, ann in enumerate(annotations, start=1):
                df_sub = smoothed_chromatogram_df[
                    smoothed_chromatogram_df["annotation"] == ann
                ]
                fig_sub.add_trace(
                    go.Scatter(
                        x=df_sub["rt"],
                        y=df_sub["intensity"],
                        mode="lines",
                        name=str(ann),
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )

                # add vertical lines for picked peaks
                peaks_sub = st.session_state.picked_peaks_df[
                    st.session_state.picked_peaks_df["annotation"] == ann
                ]

                multiple_peaks = len(peaks_sub) > 1
                colors = ["red", "blue", "green", "orange", "purple", "brown"]
                idx = 0
                for _, peak in peaks_sub.iterrows():
                    if multiple_peaks:
                        ann_text = f"Feature {idx}"
                    else:
                        ann_text = f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity']:.0f}"
                    fig_sub.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color=colors[idx], width=1, dash="dash"),
                        row=i,
                        col=1,
                        annotation_text=ann_text,
                        annotation_position="top left",
                        name=str(ann) + " Peak " + str(idx),
                    )

                    fig_sub.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color=colors[idx], width=1, dash="dash"),
                        row=i,
                        col=1,
                        showlegend=False,
                        name=str(ann) + " Peak " + str(idx),
                    )
                    idx += 1

                fig_sub.update_yaxes(title_text="Intensity", row=i, col=1)
            fig_sub.update_layout(
                height=100 * len(annotations), title_text="XICs per Transition"
            )
            progress_pp.progress(100)
            st.plotly_chart(fig_sub, use_container_width=True)

            if smoothing_method in ["Gaussian"]:
                st.markdown(
                    ":blue[**Note:** The XIC rendered is the raw data, peak picking was performed using the gaussian smoothed data internally.]"
                )

            st.markdown("""
            In the plot above, we show the XICs for each transition in separate subplots, with vertical dashed lines indicating the left and right peak boundaries identified by the peak picking algorithm. The annotation for each peak shows the FWHM and integrated intensity for that peak. Depending on the smoothing method used, the detected peaks and their boundaries may differ. We can also see that the individual peak boundaries may differ slightly across transitions. The next step is to merge these individual peak boundaries across transitions to generate consensus peak boundaries for the precursor feature.
                        """)

            # button to perform merging of transition-level peak boundaries into consensus features
            merge_clicked = st.button(
                "Merge Transition-Level Peak Boundaries into Consensus Features",
                type="primary",
            )
            if merge_clicked:
                consensus_df, members_df = (
                    merge_transition_peak_boundaries_to_consensus(
                        st.session_state.picked_peaks_df,
                        intensity_col="integrated_intensity_fda",
                        min_annotations=2,
                        keep_singletons=True,
                        boundary_mode="weighted_median",
                        min_overlap=0.05,
                        apex_tol_factor=0.4,
                        gap_factor=0.1,
                    )
                )

                st.dataframe(consensus_df)

                group_cols = ["ms_level", "annotation", "rt"]
                integrate_col = "intensity"

                # compute smoothed dataframe (used for both combined and per-transition plots)
                exp_df_targeted = st.session_state.exp_df_targeted
                smoothed_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(group_cols)[integrate_col]
                    .sum()
                    .reset_index()
                    .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                    .apply(apply_sgolay, window_length=9, polyorder=3)
                    .reset_index(drop=True)
                )

                # Get max smoothed_int across all data
                max_smoothed_int = smoothed_df["smoothed_int"].max()
                consensus_df[["apexIntensity"]] = max_smoothed_int

                xic_plot = smoothed_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="smoothed_int",
                    by="annotation",
                    title=f"Smoothed Chromatogram with Consensus Features - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=consensus_df,
                    legend_config=dict(title="Transition"),
                    # feature_config=dict(),
                    backend="ms_plotly",
                    show_plot=False,
                )

                st.plotly_chart(xic_plot, use_container_width=True)

                st.markdown("""
                In the plot above, we show the smoothed XICs for each transition with the consensus peak boundaries plotted as vertical solid lines. 
                
                But why go through all the trouble of doing individual transition-level peak picking and then merging into consensus features? Why not just perform peak picking on the summed XICs across all transitions for a precursor? Let's try that next.
                """)

                # =========================================
                #   Raw concensus with raw peak picking
                concensus_raw_chrom_df = create_concensus_chromatogram(exp_df_targeted)

                # perform peak picking
                picker = poms.PeakPickerChromatogram()
                picker_params = picker.getDefaults()
                picker_params.setValue("method", "legacy")
                picker.setParameters(picker_params)

                concensus_raw_pp_df = perform_xic_peak_picking(
                    concensus_raw_chrom_df, intensity_col="intensity", picker=picker
                )
                concensus_raw_chrom_df[["apexIntensity"]] = concensus_raw_chrom_df[
                    "intensity"
                ].max()

                concensus_raw_chrom_plot = concensus_raw_chrom_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="intensity",
                    title=f"Concensus Chromatogram with Peak Picking - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=concensus_raw_pp_df,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )

                # =========================================
                #   Raw concensus with smoothed peak picking
                picker = poms.PeakPickerChromatogram()
                picker_params = picker.getDefaults()
                picker_params.setValue("method", "corrected")
                picker_params.setValue("sgolay_frame_length", 9)
                picker_params.setValue("sgolay_polynomial_order", 3)
                picker.setParameters(picker_params)
                concensus_raw_pp_df_smooth = perform_xic_peak_picking(
                    concensus_raw_chrom_df, intensity_col="intensity", picker=picker
                )
                concensus_raw_chrom_df[["apexIntensity"]] = concensus_raw_chrom_df[
                    "intensity"
                ].max()
                concensus_raw_chrom_plot_smooth = concensus_raw_chrom_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="intensity",
                    title=f"Concensus Chromatogram with Smoothed Peak Picking - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=concensus_raw_pp_df_smooth,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )

                # =========================================
                #   smoothed concensus with smoothed peak picking
                group_cols = ["ms_level", "annotation", "rt"]
                integrate_col = "intensity"

                exp_df_targeted = st.session_state.exp_df_targeted
                smoothed_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(group_cols)[integrate_col]
                    .sum()
                    .reset_index()
                    .groupby(["annotation", "ms_level"])[group_cols + [integrate_col]]
                    .apply(apply_sgolay, window_length=9, polyorder=3)
                    .reset_index(drop=True)
                )

                picker = poms.PeakPickerChromatogram()
                picker_params = picker.getDefaults()
                picker_params.setValue("method", "corrected")
                picker_params.setValue("sgolay_frame_length", 9)
                picker_params.setValue("sgolay_polynomial_order", 3)
                picker.setParameters(picker_params)

                smoothed_for_consensus = smoothed_df[["rt", "annotation"]].copy()
                smoothed_for_consensus["intensity"] = smoothed_df["smoothed_int"]
                smoothed_concensus_chrom_df = create_concensus_chromatogram(
                    smoothed_for_consensus
                )
                smoothed_concensus_pp_df = perform_xic_peak_picking(
                    smoothed_concensus_chrom_df,
                    intensity_col="intensity",
                    picker=picker,
                )
                smoothed_concensus_chrom_df[["apexIntensity"]] = (
                    smoothed_concensus_chrom_df["intensity"].max()
                )

                smoothed_concensus_chrom_plot = smoothed_concensus_chrom_df.plot(
                    kind="chromatogram",
                    x="rt",
                    y="intensity",
                    title=f"Smoothed Concensus Chromatogram with Peak Picking - {peptide} / {precursor_charge}+",
                    aggregate_duplicates=False,
                    annotation_data=smoothed_concensus_pp_df,
                    legend_config=dict(title="Transition"),
                    backend="ms_plotly",
                    show_plot=False,
                )

                # ==================================================
                # Create a 3x2 layout: left column = concensus plots, right column = individual-transition XICs
                combined_concensus_plot = make_subplots(
                    rows=3,
                    cols=2,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.08,
                    subplot_titles=[
                        "Raw Concensus Chromatogram (Raw PP)",
                        "Raw XICs (per transition)",
                        "Raw Concensus Chromatogram (Smoothed PP)",
                        "Raw XICs (per transition)",
                        "Smoothed Concensus Chromatogram (Smoothed PP)",
                        "Smoothed XICs (per transition)",
                    ],
                )

                # prepare per-transition raw and smoothed data
                exp_df_targeted = st.session_state.exp_df_targeted
                # raw per-transition XICs (unsmoothed)
                raw_transitions_df = (
                    exp_df_targeted.apply(
                        lambda x: x.fillna(0)
                        if x.dtype.kind in "biufc"
                        else x.fillna(".")
                    )
                    .groupby(["annotation", "rt"])["intensity"]
                    .sum()
                    .reset_index()
                )

                # smoothed per-transition XICs (smoothed_df was computed earlier)
                smoothed_transitions_df = smoothed_df.copy()

                # Row 1: Raw concensus with raw peak picking (left) and raw transitions (right)
                combined_concensus_plot.add_trace(
                    go.Scatter(
                        x=concensus_raw_chrom_df["rt"],
                        y=concensus_raw_chrom_df["intensity"],
                        mode="lines",
                        name="Raw Concensus Chromatogram",
                    ),
                    row=1,
                    col=1,
                )
                for _, peak in concensus_raw_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=1,
                        annotation_text=f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity_fda']:.0f}",
                        annotation_position="top left",
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=1,
                    )

                # right column: raw per-transition traces (with legend grouping across all rows)
                annotations = list(raw_transitions_df["annotation"].unique())
                for ai, ann in enumerate(annotations):
                    df_sub = raw_transitions_df[raw_transitions_df["annotation"] == ann]
                    combined_concensus_plot.add_trace(
                        go.Scatter(
                            x=df_sub["rt"],
                            y=df_sub["intensity"],
                            mode="lines",
                            name=str(ann),
                            legendgroup=str(ann),
                            showlegend=True,
                        ),
                        row=1,
                        col=2,
                    )
                # overlay concensus boundaries on right column
                for _, peak in concensus_raw_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=2,
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="red", width=1, dash="dash"),
                        row=1,
                        col=2,
                    )

                # Row 2: Raw concensus with smoothed peak picking (left) and raw transitions (right)
                combined_concensus_plot.add_trace(
                    go.Scatter(
                        x=concensus_raw_chrom_df["rt"],
                        y=concensus_raw_chrom_df["intensity"],
                        mode="lines",
                        name="Raw Concensus Chromatogram",
                    ),
                    row=2,
                    col=1,
                )
                for _, peak in concensus_raw_pp_df_smooth.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=1,
                        annotation_text=f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity_fda']:.0f}",
                        annotation_position="top left",
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=1,
                    )

                # right column: raw per-transition traces (same groups; hide duplicate legend entries)
                for ai, ann in enumerate(annotations):
                    df_sub = raw_transitions_df[raw_transitions_df["annotation"] == ann]
                    combined_concensus_plot.add_trace(
                        go.Scatter(
                            x=df_sub["rt"],
                            y=df_sub["intensity"],
                            mode="lines",
                            name=str(ann),
                            legendgroup=str(ann),
                            showlegend=False,
                        ),
                        row=2,
                        col=2,
                    )
                for _, peak in concensus_raw_pp_df_smooth.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=2,
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="green", width=1, dash="dash"),
                        row=2,
                        col=2,
                    )

                # Row 3: Smoothed concensus with smoothed peak picking (left) and smoothed transitions (right)
                combined_concensus_plot.add_trace(
                    go.Scatter(
                        x=smoothed_concensus_chrom_df["rt"],
                        y=smoothed_concensus_chrom_df["intensity"],
                        mode="lines",
                        name="Smoothed Concensus Chromatogram",
                    ),
                    row=3,
                    col=1,
                )
                for _, peak in smoothed_concensus_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=1,
                        annotation_text=f"FWHM: {peak['FWHM']:.2f}s | Int: {peak['integrated_intensity_fda']:.0f}",
                        annotation_position="top left",
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=1,
                    )

                # right column: smoothed per-transition traces
                smoothed_annotations = list(
                    smoothed_transitions_df["annotation"].unique()
                )
                for ai, ann in enumerate(smoothed_annotations):
                    df_sub = smoothed_transitions_df[
                        smoothed_transitions_df["annotation"] == ann
                    ]
                    combined_concensus_plot.add_trace(
                        go.Scatter(
                            x=df_sub["rt"],
                            y=df_sub["smoothed_int"],
                            mode="lines",
                            name=str(ann),
                            legendgroup=str(ann),
                            showlegend=False,
                        ),
                        row=3,
                        col=2,
                    )
                for _, peak in smoothed_concensus_pp_df.iterrows():
                    combined_concensus_plot.add_vline(
                        x=peak["leftWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=2,
                    )
                    combined_concensus_plot.add_vline(
                        x=peak["rightWidth"],
                        line=dict(color="blue", width=1, dash="dash"),
                        row=3,
                        col=2,
                    )

                # axis labels and layout
                combined_concensus_plot.update_yaxes(
                    title_text="Intensity", row=1, col=1
                )
                combined_concensus_plot.update_yaxes(
                    title_text="Intensity", row=2, col=1
                )
                combined_concensus_plot.update_yaxes(
                    title_text="Intensity", row=3, col=1
                )
                combined_concensus_plot.update_layout(
                    height=900,
                    title_text="Comparison of Concensus Chromatograms and Individual Transition XICs",
                )
                st.plotly_chart(combined_concensus_plot, use_container_width=True)

                st.markdown("""
                From the plots above, we can see that performing peak picking on the summed XICs across all transitions (concensus chromatogram) can yield different peak boundaries compared to performing peak picking on the individual transitions and then merging into consensus features. This is because the peak picking algorithm may identify different peaks and boundaries when applied to the summed signal compared to the individual signals, especially if there is variability in the peak shapes and retention times across transitions. Additionally, the choice of smoothing method and parameters can also impact the detected peaks and their boundaries. It's important to carefully consider these factors when designing a DIA data processing pipeline and to validate the results using appropriate metrics and visualizations.
                """)
