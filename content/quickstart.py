"""
OpenDIAKiosk Quickstart Page.

Landing page that briefly introduces OpenDIAKiosk and offers a styled
Windows download card when the packaged executable is available.
"""

from pathlib import Path

import streamlit as st

from src.common.common import page_setup

WINDOWS_APP_PATH = Path("/app/OpenDIAKiosk.zip")


@st.cache_resource
def load_windows_app_bytes() -> bytes | None:
    if WINDOWS_APP_PATH.exists():
        return WINDOWS_APP_PATH.read_bytes()
    return None


def render_windows_download_box(app_bytes: bytes) -> None:
    """Render a styled offline-download card for the Windows app."""
    container_key = "windows_download_container"

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
                OpenDIAKiosk for Windows
            </h4>
            <p style="color: #6c757d; margin-bottom: 1rem;">
                You can download an offline version for Windows systems below.
            </p>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns([2, 3, 2])
        with cols[1]:
            st.download_button(
                label="📥 Download for Windows",
                data=app_bytes,
                file_name="OpenDIAKiosk.zip",
                mime="application/zip",
                type="secondary",
                use_container_width=True,
                help="Download OpenDIAKiosk for Windows systems",
            )

        st.markdown(
            """
            <div style="text-align: center; margin-top: 1rem; color: #6c757d;">
                Extract the zip file and run the installer (.msi) to install the app.
                Launch using the desktop icon after installation.<br>
                Even offline, it's still a web app - just packaged so you can use it without an internet connection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <style>
        .st-key-{container_key} {{
            background: linear-gradient(135deg, #f8f9fa 0%, #f1f3f4 100%) !important;
            border: 1px solid #e0e0e0 !important;
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

windows_app_bytes = load_windows_app_bytes()
if windows_app_bytes is not None:
    render_windows_download_box(windows_app_bytes)

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
