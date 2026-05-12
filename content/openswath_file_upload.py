import streamlit as st

from src.common.common import page_setup
from src.workflow.OpenSwathWorkflow import OpenSwathWorkflow


params = page_setup()

wf = OpenSwathWorkflow()
wf.show_file_upload_section()
