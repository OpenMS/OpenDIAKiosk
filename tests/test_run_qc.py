"""Per-run QC read from an OSW file (src/run_qc.py)."""

import sqlite3

import numpy as np
import pytest

from src.run_qc import (
    detections,
    detections_figure,
    identified_peaks,
    per_run_box_figure,
    rt_calibration_figure,
    rt_fit,
    rt_residual_figure,
    run_name,
    run_summary,
)


@pytest.fixture
def osw(tmp_path):
    """Two runs; precursors 1-4 are targets, 9 is a decoy."""
    path = tmp_path / "qc.osw"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE RUN (ID INT, FILENAME TEXT);
        CREATE TABLE PRECURSOR (ID INT, LIBRARY_RT REAL, DECOY INT);
        CREATE TABLE FEATURE (ID INT, RUN_ID INT, PRECURSOR_ID INT, EXP_RT REAL, NORM_RT REAL,
            DELTA_RT REAL, LEFT_WIDTH REAL, RIGHT_WIDTH REAL);
        CREATE TABLE FEATURE_MS2 (FEATURE_ID INT, AREA_INTENSITY REAL, VAR_MASSDEV_SCORE REAL);
        CREATE TABLE FEATURE_MS1 (FEATURE_ID INT, VAR_MASSDEV_SCORE REAL);
        CREATE TABLE SCORE_MS2 (FEATURE_ID INT, SCORE REAL, RANK INT, PVALUE REAL, QVALUE REAL, PEP REAL);
        INSERT INTO RUN VALUES (1, '/data/run_a.mzML'), (2, 'C:\\data\\run_b.mzML.gz');
        INSERT INTO PRECURSOR VALUES (1, 10, 0), (2, 20, 0), (3, 30, 0), (4, 40, 0), (9, 25, 1);
        -- run_a: RT = 2 * iRT + 100; precursor 4 is above the cutoff, 9 is a decoy
        INSERT INTO FEATURE VALUES
            (11, 1, 1, 120, 10.5, 0, 115, 125),
            (12, 1, 2, 140, 19.5, 0, 136, 144),
            (13, 1, 3, 160, 30.0, 0, 154, 166),
            (14, 1, 4, 180, 40.0, 0, 170, 190),
            (19, 1, 9, 150, 25.0, 0, 145, 155),
            (21, 2, 1, 121, 10.0, 0, 116, 126),
            (22, 2, 2, 141, 20.0, 0, 131, 151),
            (23, 2, 2, 300, 20.0, 0, 295, 305);
        INSERT INTO FEATURE_MS2 VALUES (11, 1, 2.0), (12, 1, 3.0), (13, 1, 4.0), (14, 1, 9.0), (19, 1, 9.0),
            (21, 5, 1.0), (22, 5, 1.5), (23, 50, 1.5);
        INSERT INTO FEATURE_MS1 VALUES (11, 5.0), (12, 6.0), (13, 7.0);
        INSERT INTO SCORE_MS2 VALUES (11, 0, 1, 0, 0.001, 0), (12, 0, 1, 0, 0.001, 0), (13, 0, 1, 0, 0.005, 0),
            (14, 0, 1, 0, 0.2, 0), (19, 0, 1, 0, 0.001, 0), (21, 0, 1, 0, 0.001, 0), (22, 0, 1, 0, 0.001, 0),
            (23, 0, 2, 0, 0.001, 0);
        """
    )
    conn.close()
    return path


def test_run_name():
    assert run_name("/data/run_a.mzML") == "run_a"
    assert run_name("C:\\data\\run_b.mzML.gz") == "run_b"


def test_identified_peaks_keeps_rank_one_targets_below_cutoff(osw):
    peaks = identified_peaks(osw, max_qvalue=0.01)
    assert sorted(peaks.loc[peaks["run"] == "run_a", "precursor"]) == [1, 2, 3]
    assert sorted(peaks.loc[peaks["run"] == "run_b", "precursor"]) == [1, 2]
    assert peaks["scored"].all()
    # the rank-2 peak at 300 s is not used
    assert peaks["rt"].max() == 160
    assert identified_peaks(osw, max_qvalue=0.5)["precursor"].nunique() == 4


def test_identified_peaks_without_scores_uses_the_largest_peak(osw):
    with sqlite3.connect(osw) as conn:
        conn.execute("DROP TABLE SCORE_MS2")
    peaks = identified_peaks(osw)
    run_b = peaks[peaks["run"] == "run_b"].set_index("precursor")
    assert run_b.loc[2, "rt"] == 300
    assert not peaks["scored"].any()
    assert 9 not in set(peaks["precursor"])


def test_summary_and_fit(osw):
    peaks = identified_peaks(osw)
    fit = rt_fit(peaks[peaks["run"] == "run_a"])
    assert fit["slope"] == pytest.approx(2.0) and fit["intercept"] == pytest.approx(100.0)
    assert fit["r2"] == pytest.approx(1.0)
    summary = run_summary(peaks).set_index("run")
    assert summary.loc["run_a", "precursors"] == 3
    assert summary.loc["run_a", "median peak width (s)"] == 10
    assert summary.loc["run_a", "median fragment error (ppm)"] == 3.0
    assert summary.loc["run_a", "median precursor error (ppm)"] == 6.0
    # two points are not enough for a fit
    assert np.isnan(summary.loc["run_b", "RT fit R²"])


def test_detections(osw):
    found = detections(identified_peaks(osw))
    assert found.to_dict("list") == {"at least n runs": [1, 2], "precursors": [3, 2]}


def test_figures_render(osw):
    peaks = identified_peaks(osw)
    assert len(rt_calibration_figure(peaks).data) == 3  # two scatters, one fit line
    assert len(detections_figure(peaks).data) == 2
    assert len(per_run_box_figure(peaks, "ms1_ppm", "t", "y").data) == 1  # run_b has no MS1 scores
    assert rt_residual_figure(peaks).layout.shapes
