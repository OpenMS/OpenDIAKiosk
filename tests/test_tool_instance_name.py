"""
Tests for the tool_instance_name functionality.

This module verifies that save_parameters correctly resolves tool instance names
to real tool names when calling create_ini, and that parameters are stored and
retrieved using the instance name as the key.
"""
import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Mock streamlit before importing ParameterManager
mock_streamlit = MagicMock()
mock_streamlit.session_state = {}

_original_streamlit = sys.modules.get('streamlit')
sys.modules['streamlit'] = mock_streamlit


def _drop_cached_workflow_modules() -> None:
    """Forget any cached src.workflow modules.

    A module binds `st` once, at import time. If an earlier test file has already
    imported src.workflow.ParameterManager against the real streamlit, the import
    below is just a cache hit and the mock never takes effect - which is why these
    tests passed when run alone but failed in a full-suite run.
    """
    for _key in list(sys.modules.keys()):
        if _key.startswith('src.workflow'):
            sys.modules.pop(_key, None)


# Drop first, so the import below really binds the mock.
_drop_cached_workflow_modules()

from src.workflow.ParameterManager import ParameterManager

if _original_streamlit is not None:
    sys.modules['streamlit'] = _original_streamlit
else:
    sys.modules.pop('streamlit', None)

# Drop again, so later test files re-import against the real streamlit instead of
# the mock-bound modules this file just created.
_drop_cached_workflow_modules()


@pytest.fixture
def temp_workflow_dir():
    """Create a temporary workflow directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workflow_dir = Path(tmpdir) / "test-workflow"
        workflow_dir.mkdir()
        ini_dir = workflow_dir / "ini"
        ini_dir.mkdir()
        yield workflow_dir


@pytest.fixture(autouse=True)
def reset_streamlit_state():
    """Reset mock streamlit session state before each test."""
    mock_streamlit.session_state.clear()
    yield


class TestSaveParametersWithInstanceName:
    """Tests for save_parameters correctly resolving tool instance names."""

    def test_save_parameters_uses_real_tool_name_for_create_ini(self, temp_workflow_dir):
        """Test that save_parameters resolves instance name to real tool name
        before calling create_ini."""
        pm = ParameterManager(temp_workflow_dir)

        # Simulate session state with an instance name (IDFilter_step1)
        # that differs from the real tool name (IDFilter)
        mock_streamlit.session_state[f"{pm.topp_param_prefix}IDFilter_step1:1:score:pep"] = 0.05
        # Register instance mapping (as input_TOPP would)
        mock_streamlit.session_state["_topp_tool_instance_map"] = {
            "IDFilter_step1": "IDFilter"
        }

        # Mock create_ini to track calls - return False (tool not found)
        # to prevent further processing that requires actual ini files
        with patch.object(pm, 'create_ini', return_value=False) as mock_create_ini:
            pm.save_parameters()

        # Verify create_ini was called with the REAL tool name, not the instance name
        mock_create_ini.assert_called_once_with("IDFilter")

    def test_save_parameters_without_instance_map_uses_tool_name_directly(self, temp_workflow_dir):
        """Test that save_parameters works normally when no instance map exists
        (backward compatibility)."""
        pm = ParameterManager(temp_workflow_dir)

        # Simulate session state with a normal tool name (no instance mapping)
        mock_streamlit.session_state[f"{pm.topp_param_prefix}IDFilter:1:score:pep"] = 0.05

        with patch.object(pm, 'create_ini', return_value=False) as mock_create_ini:
            pm.save_parameters()

        # Should use the tool name directly
        mock_create_ini.assert_called_once_with("IDFilter")

    def test_save_parameters_two_instances_store_independently(self, temp_workflow_dir):
        """Two instances of one tool read the real tool's ini and are stored
        under their own instance names."""
        import pyopenms as poms

        pm = ParameterManager(temp_workflow_dir)
        param = poms.Param()
        param.setValue("IDFilter:1:score:pep", 0.01)
        poms.ParamXMLFile().store(str(temp_workflow_dir / "ini" / "IDFilter.ini"), param)

        mock_streamlit.session_state["_topp_tool_instance_map"] = {
            "IDFilter_step1": "IDFilter",
            "IDFilter_step2": "IDFilter",
        }
        mock_streamlit.session_state[f"{pm.topp_param_prefix}IDFilter_step1:1:score:pep"] = 0.05
        mock_streamlit.session_state[f"{pm.topp_param_prefix}IDFilter_step2:1:score:pep"] = 0.01

        with patch.object(pm, "ensure_default_ini", return_value=None):
            pm.save_parameters()

        with open(pm.params_file, "r") as f:
            saved = json.load(f)

        assert saved["IDFilter_step1"] == {"score:pep": 0.05}
        # matches the ini default, so nothing is stored for it
        assert saved.get("IDFilter_step2", {}) == {}

class TestGetToppParametersWithInstanceName:
    """Tests for get_topp_parameters with tool_instance_name."""

    def test_get_topp_parameters_with_instance_name(self, temp_workflow_dir):
        """Test that get_topp_parameters uses instance name for JSON lookup."""
        pm = ParameterManager(temp_workflow_dir)

        # Create params.json with instance-keyed parameters
        params = {
            "IDFilter_step1": {
                "score:pep": 0.05
            }
        }
        with open(pm.params_file, "w") as f:
            json.dump(params, f)

        # Create a mock ini file
        ini_path = temp_workflow_dir / "ini" / "IDFilter.ini"
        ini_path.touch()

        mock_param = MagicMock()
        mock_param.keys.return_value = [b"IDFilter:1:score:pep"]
        mock_param.getValue.return_value = 0.01  # default

        with patch('pyopenms.Param', return_value=mock_param), \
             patch('pyopenms.ParamXMLFile'):
            result = pm.get_topp_parameters("IDFilter", tool_instance_name="IDFilter_step1")

        # Should return the instance-specific value
        assert result["score:pep"] == 0.05

    def test_get_topp_parameters_without_instance_name_backward_compat(self, temp_workflow_dir):
        """Test that get_topp_parameters works without instance name (backward compat)."""
        pm = ParameterManager(temp_workflow_dir)

        params = {
            "IDFilter": {
                "score:pep": 0.05
            }
        }
        with open(pm.params_file, "w") as f:
            json.dump(params, f)

        ini_path = temp_workflow_dir / "ini" / "IDFilter.ini"
        ini_path.touch()

        mock_param = MagicMock()
        mock_param.keys.return_value = [b"IDFilter:1:score:pep"]
        mock_param.getValue.return_value = 0.01

        with patch('pyopenms.Param', return_value=mock_param), \
             patch('pyopenms.ParamXMLFile'):
            result = pm.get_topp_parameters("IDFilter")

        assert result["score:pep"] == 0.05
