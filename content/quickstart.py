"""
OpenDIAKiosk Quickstart Page.

Landing page that briefly introduces OpenDIAKiosk and offers a styled
download card for the desktop app (Windows, macOS, Linux) when the latest
release has installers.
"""

import streamlit as st

from src.common.common import page_setup
from src.common.installers import installers_available, render_installer_buttons


def render_desktop_download_box() -> None:
    """Render the offline-download card: one button per platform with an installer."""
    if not installers_available():
        return
    container_key = "desktop_download_container"

    st.markdown(
        """
        <h4 style="color: #6c757d; margin-bottom: 1rem; font-size: 1.3rem; font-weight: 600; text-align: center;">
            Want to run free and open DIA analysis offline?
        </h4>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key=container_key):
        st.markdown(
            """
            <h4 style="color: #6c757d; margin-bottom: 0.75rem; font-size: 1.1rem; font-weight: 600;">
                OpenDIAKiosk desktop app
            </h4>
            <p style="color: #6c757d; margin-bottom: 1rem;">
                Download the offline version for Windows, macOS or Linux below.
            </p>
            """,
            unsafe_allow_html=True,
        )

        render_installer_buttons(app_name="OpenDIAKiosk")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 1rem; color: #6c757d;">
                Even offline, it's still a web app - just packaged so you can use it without an internet connection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <style>
        .st-key-{container_key} {{
            background: rgba(128, 128, 128, 0.08) !important;
            border: 1px solid rgba(128, 128, 128, 0.3) !important;
            border-radius: 8px !important;
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
            text-align: center !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        }}

        .st-key-{container_key} > div {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


page_setup(page="main")

st.markdown("# OpenDIAKiosk")

render_desktop_download_box()

st.markdown(
    """
OpenDIAKiosk is an interactive Streamlit app for **Data-Independent Acquisition (DIA)**
mass spectrometry. It combines hands-on teaching of DIA concepts with a practical
**OpenSWATH** analysis pipeline built on the OpenMS TOPP tools.

- 📚 **Learn** the concepts behind DIA: targeted data extraction, peak picking,
  feature scoring, and statistical validation.
- 🧪 **Build** spectral libraries with in-silico prediction, filtering, and decoy generation.
- 🔁 **Analyze** your own DIA data end-to-end with the OpenSwath workflow and
  inspect results in interactive viewers.
"""
)

st.markdown("## Get Started")
st.page_link(
    "content/dia_00_concepts.py",
    label="Learn DIA Concepts",
    icon="📚",
)
st.page_link(
    "content/openswath_file_upload.py",
    label="Analyze DIA data (OpenSwathWorkflow)",
    icon="🔁",
)
