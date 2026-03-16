from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pyopenms as poms
import pyopenms_viz  # noqa: F401  # registers plotting backends for pandas
import streamlit as st
import plotly
from plotly.subplots import make_subplots
import scipy
from scipy.spatial import cKDTree

from src.common.common import page_setup
from utils.dia_tutorial import mz_extraction_windows, filter_spectrum, reduce_spectra, annotate_filtered_spectra, apply_sgolay, msexperiment_to_dataframe, plot_3d_binned_xic_scatter

page_setup()


def _fmt_num(x: float) -> str:
    """Format numeric values with up to 6 significant digits, handling NaN."""
    try:
        if np.isfinite(x):
            return f"{x:.6g}"
    except Exception:
        pass
    return "n/a"


def _range(df: pd.DataFrame, col: str) -> str:
    """Return a human-friendly range string for numeric column `col` in `df`.

    Examples:
      - "0.123 – 1.234"
      - "n/a" if no numeric data
    """
    if col not in df.columns:
        return "n/a"
    ser = pd.to_numeric(df[col], errors="coerce").dropna()
    if ser.empty:
        return "n/a"
    mn = float(ser.min())
    mx = float(ser.max())
    if np.isclose(mn, mx):
        return _fmt_num(mn)
    return f"{_fmt_num(mn)} – {_fmt_num(mx)}"

st.title("Getting Started With DIA")
st.markdown(
    """
This page will walk you through the typical concepts for DIA data anlysis using peptide-centric targeted data extraction. We will use a small sample dataset from the [DIA PASEF Evosep dataset (PXD017703)](https://www.ebi.ac.uk/pride/archive/projects/PXD017703) published by Meier et al., 2020. This dataset contains DIA data acquired on a Bruker timsTOF Pro instrument using the Evosep One LC system. The sample file we will use is a small subset of the full experiment run (*20200505_Evosep_200SPD_SG06-16_MLHeLa_200ng_py8_S3-A1_1_2737*), containing spectra in the MS1 precursor m/z range 660-700 and RT range 130-170s. 
"""
)

st.markdown("---")
st.subheader("Setup & Imports")
st.code(
    f"""import numpy as np # v{np.__version__}
    import pandas as pd # v{pd.__version__}
import matplotlib.pyplot as plt # v{matplotlib.__version__}
import pyopenms as poms # v{poms.__version__}
import pyopenms_viz # v{pyopenms_viz.__version__}, registers MS plotting methods for pandas dataframes
from plotly.subplots import make_subplots # v{plotly.__version__}
from scipy.spatial import cKDTree # v{scipy.__version__}
# Method imports below are local modules part of the DIAapp
from utils.dia_tutorial import filter_spectra, annotate_filtered_spectra""",
    language="python",
)

st.markdown("---")
st.subheader("Load mzML Data")
mz_file = "example-data/mzML/20200505_Evosep_200SPD_SG06-16_filtered_ms1_mz_660_700_rt_130_170_with_ms2.mzML.gz"

load_clicked = st.button("Load DIA Data", type="primary")

if load_clicked:
    try:
        exp = poms.MSExperiment()
        poms.MzMLFile().load(mz_file, exp)
        exp_df = exp.to_df(long_format=True)

        st.success("Data loaded successfully.")
        st.markdown(f"Experiment summary: `MSExperiment(num_spectra={exp.getNrSpectra()}, num_chromatograms={exp.getNrChromatograms()}), ms_levels={exp.getMSLevels()}, rt_range=({exp.getMinRT()}, {exp.getMaxRT()}), mz_range=({exp.getMinMZ()}, {exp.getMaxMZ()})`")
        st.markdown("You can see our sample experiment file has 122 spectra, and contains both MS1 and MS2 spectra as expected for DIA data. We can also see the RT range is from 130.2601 to 169.9141, which matches our expected filtered RT range of 130-170s. The m/z range is from 95.0050 to 1704.9105, which is all inclusive of both the MS1 and MS2 m/z data")
        
        with st.expander("Code:"):
            st.code(
                """exp = poms.MSExperiment()
poms.MzMLFile().load(mz_file, exp)
exp_df = exp.to_df(long_format=True)
print(exp)
print(exp_df['ms_level'].drop_duplicates())
""",
                language="python",
            )

        st.markdown("---")
        st.subheader("Raw DIA Peakmaps")

        fig1 = exp_df.loc[exp_df["ms_level"] == 1].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            title="MS1 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_plotly",
        )
        fig2 = exp_df.loc[exp_df["ms_level"] == 2].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            title="MS2 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_plotly",
        )
        
        fig_all = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["MS1 Spectra", "MS2 Spectra"],
        )
        for trace in fig1.data:
            fig_all.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig_all.add_trace(trace, row=1, col=2)
            
        # Add axis labels
        fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=1)
        fig_all.update_yaxes(title_text="m/z", row=1, col=1)
        fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=2)
        fig_all.update_yaxes(title_text="m/z", row=1, col=2)
        
        st.plotly_chart(fig_all, use_container_width=True)
        
        st.markdown(":blue[**Note:** the peakmaps are z-log scaled and binned for easier and quciker visualization.]")

        
        st.markdown("Looking at the peakmaps, we can see the MS1 spectra on the left showing the detected precursor ions with an m/z range of 660 to 700. The MS2 spectra on the right show the fragment ions generated from DIA isolation windows. We can see that the MS2 spectra are more complex and contain many fragment ions across a wide m/z range, which is typical for DIA data since multiple precursors are fragmented together. The intensity of the MS2 spectra is generally lower than the MS1 spectra, which is also expected since the signal is distributed across many fragments. In the next sections, we will apply targeted extraction to pull out specific precursor and fragment ions of interest from this complex DIA data.")
        
        with st.expander("Code:"):
            st.code(
            """
fig1 = exp_df.loc[exp_df["ms_level"] == 1].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    title="MS1 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_plotly",
)
fig2 = exp_df.loc[exp_df["ms_level"] == 2].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    title="MS2 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_plotly",
)

fig_all = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["MS1 Spectra", "MS2 Spectra"],
)
for trace in fig1.data:
    fig_all.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig_all.add_trace(trace, row=1, col=2)
    
# Add axis labels
fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=1)
fig_all.update_yaxes(title_text="m/z", row=1, col=1)
fig_all.update_xaxes(title_text="Retention Time (s)", row=1, col=2)
fig_all.update_yaxes(title_text="m/z", row=1, col=2)
            """,
            language="python",
            )

        st.markdown("---")
        st.subheader("Targeted Extraction")
        
        st.markdown("""
                    For our small example file, we will perform targeted data extraction for the peptide **NTGIIC(UniMod:4)TIGPASR** with charge state 2. (The full mzML file was filtered for this specific example in mind based on prior analysis).
                    
                    To perform targeted extraction, we need some information about the precursor ion we want to extract (m/z and charge), as well as the expected fragment ions (m/z, charge, and annotation if available). 
                    
                    For our example peptide, **NTGIIC(UniMod:4)TIGPASR** with charge state 2, we expect the following precursor and fragment ions:
                    
                    **precursor m/z (charge 2)**: 680.3561
                    
                    | Fragment Ion | m/z | Charge | Annotation |
                    |--------------|---------|--------|------------|
                    | 386.2034     | 386.2034| 1      | b4^1 |
                    | 487.2623     | 487.2623| 1      | y5^1 |
                    | 600.3464     | 600.3464| 1      | y6^1 |
                    | 701.3941     | 701.3941| 1      | y7^1 |
                    | 861.4247     | 861.4247| 1      | y8^1 |
                    | 974.5088     | 974.5088| 1      | y9^1 |
                    """
        )
        
        st.markdown("We use these target m/z values as coordinates and extract peaks in spectra that fall within the specified m/z tolerance window around eash target m/z. This is done for both the precursor ion in the MS1 spectra and the fragment ions in the MS2 spectra. The result is a filtered set of spectra that contain peaks matching our target precursor and fragment ions.")
        
        st.markdown(":blue[**Note:** In a real analysis scenario, you would perform additional filtering of the spectra by retention time. You would typically have prior information of where the target peptide has eluted from the liquid chromatography column (from (pseudo)DDA experiments or from predictions). These are usually expected normalized retention time values, however, for this sampled experiment file, we already filtered the full mzML file around where we expect to see the peak eluting in retention time, so there is no need for additional retention time filtering.]")
        
        st.markdown("---")
        st.markdown("#### Filtering & Annotation")

        
        precursor_mz = 680.3561
        precursor_charge = 2
        product_mzs = [386.2034,487.2623,600.3464,701.3941,861.4247,974.5088]
        product_charges = [1,1,1,1,1,1]
        product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
        prec_mz_tol = 15
        prod_mz_tol = 20
        
        filtered_exp  = reduce_spectra(exp, float(precursor_mz), product_mzs, float(prec_mz_tol), float(prod_mz_tol))
        filtered_exp.updateRanges()
        
        st.write(f"Filtered experiment summary: `MSExperiment(num_spectra={filtered_exp.getNrSpectra()}, num_chromatograms={filtered_exp.getNrChromatograms()}), ms_levels={filtered_exp.getMSLevels()}, rt_range=({filtered_exp.getMinRT()}, {filtered_exp.getMaxRT()}), mz_range=({filtered_exp.getMinMZ()}, {filtered_exp.getMaxMZ()})`")
        
        st.write("We can see that the number of spectra in the filtered experiment has reduced from 122 to 81, which means we have extracted a subset of spectra that contain peaks matching our target precursor and fragment ions. The retention time ranges remained the same because we did not apply any RT filtering, but the m/z range has narrowed to 386.1987 to 974.5265, which is  within the m/z values of our target precursor and fragment ions.")

        exp_df_targeted = filtered_exp.to_df(long_format=True)
        
        st.dataframe(pd.concat([exp_df_targeted.head(), exp_df_targeted.tail()]),  use_container_width=True)
        
        st.markdown("We can see from the resulting filtered spectra dataframe that we have spectra with both MS1 and MS2 spectra that contain peaks matching our target precursor and fragment ions. However, at this point, the spectra are not yet annotated with which peaks correspond to which target ions. The next step is to annotate the filtered spectra with the precursor and fragment ion assignments based on which target m/z they matched. We can also calculate the mass error in ppm for each matched peak to see how close the observed m/z is to the expected m/z for each target ion.")
        
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

        st.dataframe(pd.concat([exp_df_targeted.head(), exp_df_targeted.tail()]),  use_container_width=True)
        
        with st.expander("Code:"):
            st.code(
                """precursor_mz = 680.3561
precursor_charge = 2
product_mzs = [386.2034,487.2623,600.3464,701.3941,861.4247,974.5088]
product_charges = [1,1,1,1,1,1]
product_annotations = ["b4^1", "y5^1", "y6^1", "y7^1", "y8^1", "y9^1"]
prec_mz_tol = 15
prod_mz_tol = 20

filtered_exp  = reduce_spectra(exp, float(precursor_mz), product_mzs, float(prec_mz_tol), float(prod_mz_tol))
filtered_exp.updateRanges()
exp_df_targeted = filtered_exp.to_df(long_format=True)
exp_df_targeted = annotate_filtered_spectra(
    filtered_df=exp_df_targeted,
    precursor_mz=float(precursor_mz),
    precursor_charge=int(precursor_charge),
    product_mzs=product_mzs,
    product_charges=product_charges,
    product_annotations=product_annotations,
    prec_mz_tol=float(prec_mz_tol),
    prod_mz_tol=float(prod_mz_tol),
)""",
                language="python",
            )

        st.markdown("---")
        st.markdown("#### Raw vs Filtered Peakmaps")
        fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(12, 8))
        axes_cmp = axes_cmp.flatten()

        exp_df.loc[exp_df["ms_level"] == 1].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            canvas=axes_cmp[0],
            title="Raw MS1 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_matplotlib",
        )
        exp_df.loc[exp_df["ms_level"] == 2].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            canvas=axes_cmp[1],
            title="Raw MS2 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_matplotlib",
        )
        exp_df_targeted.loc[exp_df_targeted["ms_level"] == 1].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            canvas=axes_cmp[2],
            title="Filtered MS1 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_matplotlib",
        )
        exp_df_targeted.loc[exp_df_targeted["ms_level"] == 2].plot(
            kind="peakmap",
            x="rt",
            y="mz",
            z="intensity",
            canvas=axes_cmp[3],
            title="Filtered MS2 Spectra",
            aggregate_duplicates=True,
            z_log_scale=True,
            num_x_bins=500,
            num_y_bins=500,
            show_plot=False,
            backend="ms_matplotlib",
        )
        axes_cmp[2].set_ylim(axes_cmp[0].get_ylim())
        axes_cmp[3].set_ylim(axes_cmp[1].get_ylim())
        fig_cmp.tight_layout()
        st.pyplot(fig_cmp, use_container_width=True)
        
        st.markdown("Looking at the raw vs filtered peakmaps, we can see that the filtered spectra contain a small subset of peaks that match our target precursor and fragment ions. The raw MS1 spectra show many precursor ion signals across the m/z range of 660-700, while the filtered MS1 spectra show only the peaks around our target precursor m/z of 680.3561. Similarly, the raw MS2 spectra show many fragment ions across a wide m/z range, while the filtered MS2 spectra show only the peaks around our target fragment m/z values. This illustrates how targeted extraction can pull out specific signals of interest from multiplexed spectra in DIA data.")
        
        with st.expander("Code:"):
            st.code(
                """fig_cmp, axes_cmp = plt.subplots(2, 2, figsize=(12, 8))
axes_cmp = axes_cmp.flatten()
exp_df.loc[exp_df["ms_level"] == 1].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    canvas=axes_cmp[0],
    title="Raw MS1 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_matplotlib",
)
exp_df.loc[exp_df["ms_level"] == 2].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    canvas=axes_cmp[1],
    title="Raw MS2 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_matplotlib",
)
exp_df_targeted.loc[exp_df_targeted["ms_level"] == 1].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    canvas=axes_cmp[2],
    title="Filtered MS1 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_matplotlib",
)
exp_df_targeted.loc[exp_df_targeted["ms_level"] == 2].plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    canvas=axes_cmp[3],
    title="Filtered MS2 Spectra",
    aggregate_duplicates=True,
    z_log_scale=True,
    num_x_bins=500,
    num_y_bins=500,
    show_plot=False,
    backend="ms_matplotlib",
)
axes_cmp[2].set_ylim(axes_cmp[0].get_ylim())
axes_cmp[3].set_ylim(axes_cmp[1].get_ylim())
fig_cmp.tight_layout()""",
                language="python",
            )

        with st.spinner("Generating 3D peakmaps — this may take a moment..."):
            progress = st.progress(0)
            exp_df_copy = exp_df.copy()
            # For fun, lets match the original exp_df with the targeted one to mark which spectra from the raw spectra are targeted
            # we need to compare the  mz, and ms_level to find the matching spectra in exp_df and mark them as targeted
            # Vectorized nearest-neighbour style check using a KD-tree on normalized mz coordinates. 
            exp_df_copy["is_targeted"] = False
            if not exp_df_targeted.empty:
                mz_tol = 0.01
                # process each MS level separately to ensure ms_level equality
                for ms in exp_df_copy["ms_level"].unique():
                    mask_copy = exp_df_copy["ms_level"] == ms
                    mask_target = exp_df_targeted["ms_level"] == ms
                    if not mask_target.any():
                        continue

                    tgt_mz = exp_df_targeted.loc[mask_target, "mz"].to_numpy()
                    # normalize m/z by tolerance so a radius=1 query matches the window
                    tgt_coords = (tgt_mz / mz_tol).reshape(-1, 1)
                    tree = cKDTree(tgt_coords)

                    copy_mz = exp_df_copy.loc[mask_copy, "mz"].to_numpy()
                    copy_coords = (copy_mz / mz_tol).reshape(-1, 1)

                    # find any target within the normalized radius 1.0
                    neighbors = tree.query_ball_point(copy_coords, r=1.0)
                    has_match = np.array([len(n) > 0 for n in neighbors])
                    exp_df_copy.loc[mask_copy, "is_targeted"] = has_match
            progress.progress(30)

            # Labels for legend
            exp_df_copy["is_targeted_label"] = np.where(exp_df_copy["is_targeted"], "Targeted", "Raw")
            exp_df_targeted["ms_level_label"] = exp_df_targeted["ms_level"].map({1: "MS1", 2: "MS2"}).fillna(exp_df_targeted["ms_level"].astype(str))
            progress.progress(50)

            raw_3d = exp_df_copy.plot(
                kind="peakmap",
                x="rt",
                y="mz",
                z="intensity",
                by="is_targeted_label",
                title="Raw Spectra (MS1 + MS2)",
                plot_3d=True,
                aggregate_duplicates=False,
                bin_peaks=True,
                num_x_bins=500,
                num_y_bins=500,
                aggregation_method="sum",
                show_plot=False,
                width=800,
                height=600,
                legend_config=dict(title="Targeted"),
                backend="ms_plotly",
            )
            progress.progress(70)

            filt_3d = exp_df_targeted.plot(
                kind="peakmap",
                x="rt",
                y="mz",
                z="intensity",
                by="ms_level_label",
                title="Filtered Spectra (MS1 + MS2)",
                plot_3d=True,
                aggregate_duplicates=False,
                bin_peaks=True,
                num_x_bins=500,
                num_y_bins=500,
                aggregation_method="sum",
                show_plot=False,
                width=800,
                height=600,
                legend_config=dict(title="MS Level"),
                backend="ms_plotly",
            )
            progress.progress(90)

            fig_3d = make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "scene"}, {"type": "scene"}]],
                subplot_titles=["Raw Spectra", "Filtered Spectra"],
            )
            for idx, panel in enumerate([raw_3d, filt_3d]):
                for trace in panel.data:
                    fig_3d.add_trace(trace, row=1, col=idx + 1)
                scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
                if getattr(panel.layout, "scene", None) is not None:
                    fig_3d.layout[scene_name].update(panel.layout.scene)

            fig_3d.update_layout(
                scene=dict(domain=dict(x=[0.03, 0.48])),
                scene2=dict(domain=dict(x=[0.52, 0.97])),
                width=1200,
                height=600,
                margin=dict(l=20, r=20, t=60, b=20),
            )
            fig_3d.layout.annotations[0].x = 0.25
            fig_3d.layout.annotations[1].x = 0.75
            progress.progress(100)

            st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown("We can look at the peakmaps in 3D to visually see topological features more easily. If we look at the raw spectra, which we have color coded the corresponding targeted spectra matching out charged peptide of interest (red), we can see how small the spectral peaks are for our target peptide in comparison to the rest of the spectral signal in the data. If we look at the filtered spectra, you can more clearly see matching spectral patterns of the chromatography features (i.e. the eluting chromatographic peak in retention time). You can also tell the more intense precursor ion signal (blue) apart from the much lower intensity fragment ions (red).")

        st.markdown(":orange[**Tip:** You can toggle the legend items to hide/show the targeted spectra or the MS1 vs MS2 spectra.]")
        
        with st.expander("Code:"):
            st.code(
                """exp_df_copy = exp_df.copy()
# For fun, lets match the original exp_df with the targeted one to mark which spectra from the raw spectra are targeted
# we need to compare the  mz, and ms_level to find the matching spectra in exp_df and mark them as targeted
# Vectorized nearest-neighbour style check using a KD-tree on normalized mz coordinates. 
exp_df_copy["is_targeted"] = False
if not exp_df_targeted.empty:
    mz_tol = 0.01
    # process each MS level separately to ensure ms_level equality
    for ms in exp_df_copy["ms_level"].unique():
        mask_copy = exp_df_copy["ms_level"] == ms
        mask_target = exp_df_targeted["ms_level"] == ms
        if not mask_target.any():
            continue

        tgt_mz = exp_df_targeted.loc[mask_target, "mz"].to_numpy()
        # normalize m/z by tolerance so a radius=1 query matches the window
        tgt_coords = (tgt_mz / mz_tol).reshape(-1, 1)
        tree = cKDTree(tgt_coords)

        copy_mz = exp_df_copy.loc[mask_copy, "mz"].to_numpy()
        copy_coords = (copy_mz / mz_tol).reshape(-1, 1)

        # find any target within the normalized radius 1.0
        neighbors = tree.query_ball_point(copy_coords, r=1.0)
        has_match = np.array([len(n) > 0 for n in neighbors])
        exp_df_copy.loc[mask_copy, "is_targeted"] = has_match

# Labels for legend
exp_df_copy["is_targeted_label"] = np.where(exp_df_copy["is_targeted"], "Targeted", "Raw")
exp_df_targeted["ms_level_label"] = exp_df_targeted["ms_level"].map({1: "MS1", 2: "MS2"}).fillna(exp_df_targeted["ms_level"].astype(str))

raw_3d = exp_df_copy.plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    by="is_targeted_label",
    title="Raw Spectra (MS1 + MS2)",
    plot_3d=True,
    aggregate_duplicates=False,
    bin_peaks=True,
    num_x_bins=500,
    num_y_bins=500,
    aggregation_method="sum",
    show_plot=False,
    width=800,
    height=600,
    legend_config=dict(title="Targeted"),
    backend="ms_plotly",
)

filt_3d = exp_df_targeted.plot(
    kind="peakmap",
    x="rt",
    y="mz",
    z="intensity",
    by="ms_level_label",
    title="Filtered Spectra (MS1 + MS2)",
    plot_3d=True,
    aggregate_duplicates=False,
    bin_peaks=True,
    num_x_bins=500,
    num_y_bins=500,
    aggregation_method="sum",
    show_plot=False,
    width=800,
    height=600,
    legend_config=dict(title="MS Level"),
    backend="ms_plotly",
)

fig_3d = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=["Raw Spectra", "Filtered Spectra"],
)
for idx, panel in enumerate([raw_3d, filt_3d]):
    for trace in panel.data:
        fig_3d.add_trace(trace, row=1, col=idx + 1)
    scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
    if getattr(panel.layout, "scene", None) is not None:
        fig_3d.layout[scene_name].update(panel.layout.scene)

fig_3d.update_layout(
    scene=dict(domain=dict(x=[0.03, 0.48])),
    scene2=dict(domain=dict(x=[0.52, 0.97])),
    width=1200,
    height=600,
    margin=dict(l=20, r=20, t=60, b=20),
)
fig_3d.layout.annotations[0].x = 0.25
fig_3d.layout.annotations[1].x = 0.75
                """,
                language="python",
            )

        st.markdown("---")
        st.markdown("#### Extracted Ion Chromatograms")
        
        chrom_fig = exp_df_targeted.plot(
            kind="chromatogram",
            x="rt",
            y="intensity",
            by="annotation",
            title="Chromatogram",
            aggregate_duplicates=True,
            legend_config=dict(title="Transition"),
            backend="ms_plotly",
            show_plot=False,
        )
        
        group_cols=['ms_level', 'annotation', 'rt']
        integrate_col = 'intensity'
        smoothed_chrom_fig = exp_df_targeted.apply(
                lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
            ) \
        .groupby(group_cols)[integrate_col] \
        .sum() \
        .reset_index() \
        .groupby(['annotation', 'ms_level'])[group_cols + [integrate_col]] \
        .apply(apply_sgolay, 
                window_length=9,
                polyorder=5) \
        .reset_index(drop=True) \
        .plot(
            kind="chromatogram",
            x="rt",
            y="smoothed_int",
            by="annotation",
            title="Smoothed Chromatogram",
            aggregate_duplicates=False,
            legend_config=dict(title="Transition"),
            backend="ms_plotly",
            show_plot=False,
        )
        
        chrom_subfig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Raw Chromatogram", "Smoothed Chromatogram"],
        )
        # Pair traces from the raw and smoothed chromatogram figures so that
        # legend selection is shared. We assign the same `name` and `legendgroup`
        # to corresponding traces and only show the legend entry once (left panel).
        for i, (t_raw, t_smooth) in enumerate(zip(chrom_fig.data, smoothed_chrom_fig.data)):
            name = getattr(t_raw, "name", None) or getattr(t_smooth, "name", None) or f"trace_{i}"
            t_raw.name = name
            t_raw.legendgroup = name
            t_raw.showlegend = True

            t_smooth.name = name
            t_smooth.legendgroup = name
            t_smooth.showlegend = False

            chrom_subfig.add_trace(t_raw, row=1, col=1)
            chrom_subfig.add_trace(t_smooth, row=1, col=2)
        # If one figure has extra traces, add them without grouping
        if len(chrom_fig.data) > len(smoothed_chrom_fig.data):
            for extra in chrom_fig.data[len(smoothed_chrom_fig.data) :]:
                chrom_subfig.add_trace(extra, row=1, col=1)
        elif len(smoothed_chrom_fig.data) > len(chrom_fig.data):
            for extra in smoothed_chrom_fig.data[len(chrom_fig.data) :]:
                chrom_subfig.add_trace(extra, row=1, col=2)
        chrom_subfig.update_layout(
            width=1200,
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(chrom_subfig, use_container_width=True)
        
        st.markdown("The chromatograms show the intensity of the precursor and fragment ions over retention time. The raw chromatogram on the left shows the original signal, which can be a little noisy. The smoothed chromatogram on the right applies a Savitzky-Golay filter to reduce noise and make it easier to see the elution profiles of the ions, as well as making downstream peak-picking more stable.")
        st.markdown(":orange[**Tip:** You can toggle the legend items to show/hide specific transitions and see how they co-elute, which is important for confirming the presence of the target peptide in DIA data.]")
         
        with st.expander("Code:"):
            st.code(
                """chrom_fig = exp_df_targeted.plot(
    kind="chromatogram",
    x="rt",
    y="intensity",
    by="annotation",
    title="Chromatogram",
    aggregate_duplicates=True,
    legend_config=dict(title="Transition"),
    backend="ms_plotly",
    show_plot=False,
)

group_cols=['ms_level', 'annotation', 'rt']
integrate_col = 'intensity'
smoothed_chrom_fig = exp_df_targeted.apply(
        lambda x: x.fillna(0) if x.dtype.kind in "biufc" else x.fillna(".")
    ) \
.groupby(group_cols)[integrate_col] \
.sum() \
.reset_index() \
.groupby(['annotation', 'ms_level'])[group_cols + [integrate_col]] \
.apply(apply_sgolay, 
        window_length=9,
        polyorder=5) \
.reset_index(drop=True) \
.plot(
    kind="chromatogram",
    x="rt",
    y="smoothed_int",
    by="annotation",
    title="Smoothed Chromatogram",
    aggregate_duplicates=False,
    legend_config=dict(title="Transition"),
    backend="ms_plotly",
    show_plot=False,
)

chrom_subfig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Raw Chromatogram", "Smoothed Chromatogram"],
)
# Pair traces from the raw and smoothed chromatogram figures so that
# legend selection is shared. We assign the same `name` and `legendgroup`
# to corresponding traces and only show the legend entry once (left panel).
for i, (t_raw, t_smooth) in enumerate(zip(chrom_fig.data, smoothed_chrom_fig.data)):
    name = getattr(t_raw, "name", None) or getattr(t_smooth, "name", None) or f"trace_{i}"
    t_raw.name = name
    t_raw.legendgroup = name
    t_raw.showlegend = True

    t_smooth.name = name
    t_smooth.legendgroup = name
    t_smooth.showlegend = False

    chrom_subfig.add_trace(t_raw, row=1, col=1)
    chrom_subfig.add_trace(t_smooth, row=1, col=2)
# If one figure has extra traces, add them without grouping
if len(chrom_fig.data) > len(smoothed_chrom_fig.data):
    for extra in chrom_fig.data[len(smoothed_chrom_fig.data) :]:
        chrom_subfig.add_trace(extra, row=1, col=1)
elif len(smoothed_chrom_fig.data) > len(chrom_fig.data):
    for extra in smoothed_chrom_fig.data[len(chrom_fig.data) :]:
        chrom_subfig.add_trace(extra, row=1, col=2)
chrom_subfig.update_layout(
    width=1200,
    height=500,
    margin=dict(l=20, r=20, t=60, b=20),
)
                """,
                language="python",
            )
        
        st.markdown("---")
        st.subheader("What about the Ion Mobility Dimension?")
        
        ion_df = msexperiment_to_dataframe(exp)
        ms1_ion_df = ion_df.loc[ion_df["ms_level"] == 1]
        ms2_ion_df = ion_df.loc[ion_df["ms_level"] == 2]
        
        
        st.write("MS1 Spectra")
        ms1_table = (
            "| Metric | Range |\n"
            "|---|---:|\n"
            f"| Retention time | {_range(ms1_ion_df, 'rt')} |\n"
            f"| m/z | {_range(ms1_ion_df, 'mz')} |\n"
            f"| Ion mobility | {_range(ms1_ion_df, 'ion_mobility')} |\n"
        )
        st.markdown(ms1_table)

        st.write("MS2 Spectra")
        ms2_table = (
            "| Metric | Range |\n"
            "|---|---:|\n"
            f"| Retention time | {_range(ms2_ion_df, 'rt')} |\n"
            f"| m/z | {_range(ms2_ion_df, 'mz')} |\n"
            f"| Ion mobility | {_range(ms2_ion_df, 'ion_mobility')} |\n"
        )
        st.markdown(ms2_table)

        st.markdown("""This example DIA dataset was acquired on a timsTOF instrument, which means there is an additional ion mobility dimension in the data. Our sampled dataset contains MS1 spectra containing inversed ion mobility in the range 0.60199 to 1.59998, and MS2 spectra contained inversed ion mobility in the range 0.89033 to 1.10950.""")
        
        with st.spinner("Generating 3D ion-mobility peakmaps — this may take a moment..."):
            progress = st.progress(0)

            fig1, binned_df = plot_3d_binned_xic_scatter(
                ms1_ion_df,
                bins=(100, 100, 50),
                intensity_agg="mean",
                log_color=True,
                title="MS1 Peakmap with Ion Mobility",
            )
            progress.progress(30)

            fig2, binned_df = plot_3d_binned_xic_scatter(
                ms2_ion_df,
                bins=(100, 100, 50),
                intensity_agg="mean",
                log_color=True,
                title="MS2 Peakmap with Ion Mobility",
            )
            progress.progress(60)

            # Combine the two figures into subplots, pairing traces so the legend
            # is shown once and toggles affect both panels. Also set axis titles.
            fig = make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "scene"}, {"type": "scene"}]],
                subplot_titles=["MS1 Spectra", "MS2 Spectra"],
            )
            progress.progress(70)

            n1 = len(fig1.data)
            n2 = len(fig2.data)
            max_n = max(n1, n2)
            for i in range(max_n):
                t1 = fig1.data[i] if i < n1 else None
                t2 = fig2.data[i] if i < n2 else None

                if t1 is not None:
                    # Ensure a name exists for legend grouping
                    name = getattr(t1, "name", None) or f"trace_{i}"
                    t1.name = name
                    t1.legendgroup = name
                    t1.showlegend = True
                    fig.add_trace(t1, row=1, col=1)

                if t2 is not None:
                    # Use the same legendgroup/name so toggling uses a single legend
                    name2 = getattr(t2, "name", None) or (getattr(t1, "name", None) if t1 is not None else f"trace_{i}")
                    t2.name = name2
                    t2.legendgroup = name2
                    # hide duplicate legend entry for the right panel
                    t2.showlegend = False
                    fig.add_trace(t2, row=1, col=2)

            progress.progress(90)

            # Axis labels for both 3D scenes
            fig.update_layout(
                scene=dict(xaxis_title="Retention Time (s)", yaxis_title="m/z", zaxis_title="Ion mobility"),
                scene2=dict(xaxis_title="Retention Time (s)", yaxis_title="m/z", zaxis_title="Ion mobility"),
                legend=dict(title="Transition"),
            )

            progress.progress(100)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""Looking at the spectra in 3D with the ion mobility dimension, we can see just how much more complicated the data looks like. It just looks like blobs of color""")
        
        
        st.markdown("---")
        st.markdown("#### Filtering Spectra Including Ion Mobility")
        
        
        st.markdown("""Like m/z and retention time filtering, we can further filter the spectra by ion mobility as well. This means we need to have another coordinate to extract and filter around a targeted ion mobility value. For our target peptide of interest, from prior analysis, the targeted ion mobility value is **0.96756938061**. We can include this target ion mobility in our filtering criteria using a tolerance window of 0.06.""")
        
        target_im = 0.96756938061
        im_tol = 0.06        
        filtered_exp_with_im  = reduce_spectra(exp, float(precursor_mz), product_mzs, float(prec_mz_tol), float(prod_mz_tol), float(target_im), float(im_tol))
        filtered_exp_with_im.updateRanges()
        exp_df_targeted_with_im = msexperiment_to_dataframe(filtered_exp_with_im)
        exp_df_targeted_with_im = annotate_filtered_spectra(
            filtered_df=exp_df_targeted_with_im,
            precursor_mz=float(precursor_mz),
            precursor_charge=int(precursor_charge),
            product_mzs=product_mzs,
            product_charges=product_charges,
            product_annotations=product_annotations,
            prec_mz_tol=float(prec_mz_tol),
            prod_mz_tol=float(prod_mz_tol),
        )
        
        ion_df_filtered = msexperiment_to_dataframe(filtered_exp_with_im)
        st.write(print(ion_df_filtered.head()))
        
        with st.spinner("Generating 3D ion-mobility peakmaps — this may take a moment..."):
            progress = st.progress(0)

            fig1, binned_df = plot_3d_binned_xic_scatter(
                ion_df,
                bins=(100, 100, 50),
                intensity_agg="mean",
                log_color=True,
                title="Raw Peakmap with Ion Mobility",
            )
            progress.progress(30)

            fig2, binned_df = plot_3d_binned_xic_scatter(
                ion_df_filtered,
                bins=(100, 100, 50),
                intensity_agg="mean",
                log_color=True,
                title="Filtered Peakmap with Ion Mobility",
            )
            progress.progress(60)

            # Combine the two figures into subplots, pairing traces so the legend
            # is shown once and toggles affect both panels. Also set axis titles.
            fig = make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "scene"}, {"type": "scene"}]],
                subplot_titles=["Raw Spectra", "Filtered Spectra"],
            )
            progress.progress(70)

            n1 = len(fig1.data)
            n2 = len(fig2.data)
            max_n = max(n1, n2)
            for i in range(max_n):
                t1 = fig1.data[i] if i < n1 else None
                t2 = fig2.data[i] if i < n2 else None

                if t1 is not None:
                    # Ensure a name exists for legend grouping
                    name = getattr(t1, "name", None) or f"trace_{i}"
                    t1.name = name
                    t1.legendgroup = name
                    t1.showlegend = True
                    fig.add_trace(t1, row=1, col=1)

                if t2 is not None:
                    # Use the same legendgroup/name so toggling uses a single legend
                    name2 = getattr(t2, "name", None) or (getattr(t1, "name", None) if t1 is not None else f"trace_{i}")
                    t2.name = name2
                    t2.legendgroup = name2
                    # hide duplicate legend entry for the right panel
                    t2.showlegend = False
                    fig.add_trace(t2, row=1, col=2)

            progress.progress(90)

            # Axis labels for both 3D scenes
            fig.update_layout(
                scene=dict(xaxis_title="Retention Time (s)", yaxis_title="m/z", zaxis_title="Ion mobility"),
                scene2=dict(xaxis_title="Retention Time (s)", yaxis_title="m/z", zaxis_title="Ion mobility"),
                legend=dict(title="Transition"),
            )

            progress.progress(100)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to run DIA tutorial workflow: {e}")
        
st.markdown("---")

st.subheader("Functions Used in This Tutorial")
with st.expander("Click to view code for utility functions used in the tutorial"):
    st.code(inspect.getsource(mz_extraction_windows), language="python")
    st.code(inspect.getsource(filter_spectrum), language="python")
    st.code(inspect.getsource(reduce_spectra), language="python")
    st.code(inspect.getsource(annotate_filtered_spectra), language="python")
    st.code(inspect.getsource(apply_sgolay), language="python")