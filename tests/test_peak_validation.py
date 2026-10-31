"""Skyline-style peak validation views built from an OSW file (src/peak_validation.py)."""

import math
import sqlite3

import plotly.graph_objects as go
import pytest

from src.peak_validation import (
    chosen_peaks,
    dotp,
    library_match_frame,
    peak_areas_figure,
    peak_validation_tables,
    retention_times_figure,
    shade_peak,
    short_run_names,
)


@pytest.fixture
def osw(tmp_path):
    """Two runs of one precursor with three detecting transitions (plus one identifying)."""
    path = tmp_path / "test.osw"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE RUN (ID INT, FILENAME TEXT);
        CREATE TABLE TRANSITION (ID INT, TRAML_ID TEXT, PRODUCT_MZ REAL, CHARGE INT, TYPE TEXT,
            ANNOTATION TEXT, ORDINAL INT, DETECTING INT, IDENTIFYING INT, QUANTIFYING INT,
            LIBRARY_INTENSITY REAL, DECOY INT);
        CREATE TABLE TRANSITION_PRECURSOR_MAPPING (TRANSITION_ID INT, PRECURSOR_ID INT);
        CREATE TABLE FEATURE (ID INT, RUN_ID INT, PRECURSOR_ID INT, EXP_RT REAL, NORM_RT REAL,
            DELTA_RT REAL, LEFT_WIDTH REAL, RIGHT_WIDTH REAL);
        CREATE TABLE FEATURE_MS2 (FEATURE_ID INT, AREA_INTENSITY REAL, VAR_LIBRARY_DOTPROD REAL);
        CREATE TABLE FEATURE_TRANSITION (FEATURE_ID INT, TRANSITION_ID INT, AREA_INTENSITY REAL);
        CREATE TABLE SCORE_MS2 (FEATURE_ID INT, SCORE REAL, RANK INT, PVALUE REAL, QVALUE REAL, PEP REAL);
        INSERT INTO RUN VALUES (10, 'a.mzML'), (20, 'b.mzML');
        INSERT INTO TRANSITION VALUES
            (1, 't1', 300.1, 1, 'y', 'y3', 3, 1, 0, 1, 100, 0),
            (2, 't2', 400.2, 1, 'y', NULL, 4, 1, 0, 1, 50, 0),
            (3, 't3', 500.3, 2, 'b', 'b5^2', 5, 1, 0, 1, 25, 0),
            (4, 't4', 600.4, 1, 'y', 'y6', 6, 0, 1, 0, 10, 0);
        INSERT INTO TRANSITION_PRECURSOR_MAPPING VALUES (1, 7), (2, 7), (3, 7), (4, 7);
        -- run 10: the rank-1 peak (101) is smaller than the rank-2 one (102)
        INSERT INTO FEATURE VALUES
            (101, 10, 7, 60.0, 0, -2.0, 55.0, 66.0),
            (102, 10, 7, 90.0, 0, 28.0, 85.0, 95.0),
            (201, 20, 7, 63.0, 0, 1.0, 58.0, 70.0);
        INSERT INTO FEATURE_MS2 VALUES (101, 175, 0.99), (102, 900, 0.2), (201, 300, 0.98);
        INSERT INTO SCORE_MS2 VALUES (101, 5, 1, 0, 0.001, 0), (102, 1, 2, 0, 0.4, 0), (201, 4, 1, 0, 0.002, 0);
        INSERT INTO FEATURE_TRANSITION VALUES
            (101, 1, 100), (101, 2, 50), (101, 3, 25), (101, 4, 5),
            (201, 1, 50), (201, 2, 100), (201, 3, 150), (201, 4, 9);
        """
    )
    conn.close()
    return path


def test_dotp_matches_skyline_definition():
    assert dotp([100, 50, 25], [100, 50, 25]) == pytest.approx(1.0)
    assert dotp([4, 0], [0, 9]) == pytest.approx(0.0)
    # cosine of the square roots
    assert dotp([4, 1], [1, 4]) == pytest.approx((2 * 1 + 1 * 2) / 5)
    assert math.isnan(dotp([0, 0], [1, 2]))


def test_chosen_peak_is_rank_one_not_the_largest(osw):
    peaks = chosen_peaks(osw, 7)
    assert peaks.set_index("RUN_ID")["FEATURE_ID"].to_dict() == {10: 101, 20: 201}
    assert peaks.set_index("RUN_ID").loc[10, "EXPECTED_RT"] == pytest.approx(62.0)


def test_chosen_peak_without_scores_is_the_largest(osw):
    with sqlite3.connect(osw) as conn:
        conn.execute("DROP TABLE SCORE_MS2")
    peaks = chosen_peaks(osw, 7)
    assert peaks.set_index("RUN_ID").loc[10, "FEATURE_ID"] == 102
    assert peaks["QVALUE"].isna().all()


def test_validation_tables(osw):
    runs, areas = peak_validation_tables(osw, 7, {20: "B", 10: "A"}, {2: "y4"})
    # display order follows run_labels
    assert runs["run"].tolist() == ["B", "A"]
    a = runs.set_index("run").loc["A"]
    assert (a["start (s)"], a["end (s)"], a["width (s)"]) == (55.0, 66.0, 11.0)
    assert a["q-value"] == pytest.approx(0.001)
    assert a["dotp"] == pytest.approx(1.0)
    assert runs.set_index("run").loc["B", "dotp"] < 0.95

    # only detecting transitions; the XIC name wins, the type+ordinal fallback fills gaps
    assert sorted(areas["transition"].unique()) == ["b5^2", "y3", "y4"]
    shares = areas.groupby("run")["share"].sum()
    assert shares.to_dict() == pytest.approx({"A": 100.0, "B": 100.0})


def test_validation_tables_without_mapped_runs(osw):
    runs, areas = peak_validation_tables(osw, 7, {99: "missing"})
    assert runs.empty and areas.empty


def test_library_match_frame_scales_both_sides(osw):
    _, areas = peak_validation_tables(osw, 7, {10: "A", 20: "B"})
    frame = library_match_frame(areas)
    assert set(frame["side"]) == {"A", "B", "library"}
    for _, side in frame.groupby("side"):
        assert side["relative"].max() == pytest.approx(100.0)
    library = frame[frame["side"] == "library"].set_index("label")["relative"]
    assert library.to_dict() == pytest.approx({"y3": 100.0, "y4": 50.0, "b5^2": 25.0})
    assert frame["peak_id"].is_unique and frame["matched"].all()


def test_shade_peak_fills_each_trace_inside_the_boundaries():
    fig = go.Figure(
        [
            go.Scatter(x=list(range(10)), y=[1] * 10, mode="lines", name="y3", line={"color": "#ff0000"}),
            go.Scatter(x=list(range(10)), y=[2] * 10, mode="lines", name="y4"),
        ]
    )
    shade_peak(fig, 3, 6, apex=4.5)
    fills = [t for t in fig.data if t.fill == "tozeroy"]
    assert len(fills) == 2
    assert list(fills[0].x) == [3, 4, 5, 6]
    assert fills[0].fillcolor == "rgba(255,0,0,0.3)"
    # a fill hides with its transition
    assert fills[0].legendgroup == fig.data[0].legendgroup == "y3"
    assert any(shape.x0 == 4.5 for shape in fig.layout.shapes)


def test_figures(osw):
    runs, areas = peak_validation_tables(osw, 7, {10: "A", 20: "B"})
    bars = peak_areas_figure(areas, normalized=True)
    assert bars.layout.barmode == "stack" and len(bars.data) == 3
    rt = retention_times_figure(runs)
    assert list(rt.data[0].base) == [55.0, 58.0]


def test_short_run_names():
    assert short_run_names(["r/S_A_rep1_results.xic", "r/S_B_results.xic"]) == {
        "r/S_A_rep1_results.xic": "S_A_rep1",
        "r/S_B_results.xic": "S_B",
    }
    assert short_run_names(["A1_x.xic", "B1_x.xic"]) == {"A1_x.xic": "A1", "B1_x.xic": "B1"}
    # identical stems fall back to the full paths
    assert short_run_names(["a/r.xic", "b/r.xic"]) == {"a/r.xic": "a/r.xic", "b/r.xic": "b/r.xic"}
