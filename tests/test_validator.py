import numpy as np
import pandas as pd
import pytest

from st_dssm.validator import DataValidationError, validate_graph, validate_time_series


@pytest.fixture
def valid_ts_data():
    dates = pd.date_range("2026-08-13 00:00:00", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        np.random.rand(10, 3),
        index=dates,
        columns=["A", "B", "C"]
    )
    return df, ["A", "B", "C"]


def test_validate_time_series_success(valid_ts_data):
    df, sensors = valid_ts_data
    validate_time_series(df, sensors)


def test_validate_time_series_wrong_sensor_count(valid_ts_data):
    df, _sensors = valid_ts_data
    with pytest.raises(DataValidationError, match="Expected 4 sensors"):
        validate_time_series(df, ["A", "B", "C", "D"])


def test_validate_time_series_wrong_sensor_order(valid_ts_data):
    df, _sensors = valid_ts_data
    with pytest.raises(DataValidationError, match="exactly match"):
        validate_time_series(df, ["A", "C", "B"])


def test_validate_time_series_not_datetimeindex(valid_ts_data):
    df, sensors = valid_ts_data
    df.index = range(10)
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        validate_time_series(df, sensors)


def test_validate_time_series_no_timezone(valid_ts_data):
    df, sensors = valid_ts_data
    df.index = df.index.tz_localize(None)
    with pytest.raises(DataValidationError, match="explicit timezone"):
        validate_time_series(df, sensors)


def test_validate_time_series_not_increasing(valid_ts_data):
    df, sensors = valid_ts_data
    dates = list(df.index)
    dates[1], dates[2] = dates[2], dates[1]
    df.index = dates
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_time_series(df, sensors)


def test_validate_time_series_duplicates(valid_ts_data):
    df, sensors = valid_ts_data
    dates = list(df.index)
    dates[1] = dates[0]
    df.index = dates
    with pytest.raises(DataValidationError, match="duplicates"):
        validate_time_series(df, sensors)


def test_validate_time_series_wrong_cadence(valid_ts_data):
    df, sensors = valid_ts_data
    dates = list(df.index)
    dates[2] = dates[1] + pd.Timedelta(minutes=10)
    # The rest are shifted by 5 min relative to the new dates[2] so we don't have duplicates
    for i in range(3, 10):
        dates[i] = dates[i-1] + pd.Timedelta(minutes=5)
    df.index = dates
    with pytest.raises(DataValidationError, match="non-five-minute intervals"):
        validate_time_series(df, sensors)

def test_validate_time_series_dst_gap(valid_ts_data):
    df, sensors = valid_ts_data
    dates = list(df.index)
    
    start_time = pd.Timestamp("2017-03-12 01:45:00+00:00")
    for i in range(2):
        dates[i] = start_time + pd.Timedelta(minutes=5*i)
        
    dates[2] = pd.Timestamp("2017-03-12 01:55:00+00:00")
    dates[3] = pd.Timestamp("2017-03-12 03:00:00+00:00")
    for i in range(4, 10):
        dates[i] = dates[i-1] + pd.Timedelta(minutes=5)
        
    df.index = dates
    # Should raise because we removed the DST gap exception
    with pytest.raises(DataValidationError, match="non-five-minute intervals"):
        validate_time_series(df, sensors)


def test_validate_time_series_non_numeric(valid_ts_data):
    df, sensors = valid_ts_data
    df["A"] = "string"
    with pytest.raises(DataValidationError, match="numeric"):
        validate_time_series(df, sensors)


def test_validate_time_series_infinity(valid_ts_data):
    df, sensors = valid_ts_data
    df.iloc[0, 0] = np.inf
    with pytest.raises(DataValidationError, match="infinity"):
        validate_time_series(df, sensors)


@pytest.fixture
def valid_graph_data():
    adj_mx = np.eye(3)
    graph_ids = ["A", "B", "C"]
    ts_ids = ["A", "B", "C"]
    return adj_mx, graph_ids, ts_ids


def test_validate_graph_success(valid_graph_data):
    adj_mx, graph_ids, ts_ids = valid_graph_data
    validate_graph(adj_mx, graph_ids, ts_ids)


def test_validate_graph_wrong_shape(valid_graph_data):
    adj_mx, graph_ids, ts_ids = valid_graph_data
    adj_mx = np.eye(4)
    with pytest.raises(DataValidationError, match="shape"):
        validate_graph(adj_mx, graph_ids, ts_ids)


def test_validate_graph_non_numeric(valid_graph_data):
    adj_mx, graph_ids, ts_ids = valid_graph_data
    adj_mx = np.array([["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]])
    with pytest.raises(DataValidationError, match="numeric"):
        validate_graph(adj_mx, graph_ids, ts_ids)


def test_validate_graph_non_finite(valid_graph_data):
    adj_mx, graph_ids, ts_ids = valid_graph_data
    adj_mx[0, 0] = np.inf
    with pytest.raises(DataValidationError, match="non-finite"):
        validate_graph(adj_mx, graph_ids, ts_ids)


def test_validate_graph_duplicates(valid_graph_data):
    adj_mx, graph_ids, ts_ids = valid_graph_data
    graph_ids = ["A", "A", "C"]
    with pytest.raises(DataValidationError, match="duplicates"):
        validate_graph(adj_mx, graph_ids, ts_ids)


def test_validate_graph_alignment(valid_graph_data):
    adj_mx, graph_ids, ts_ids = valid_graph_data
    ts_ids = ["A", "C", "B"]
    with pytest.raises(DataValidationError, match="exactly match"):
        validate_graph(adj_mx, graph_ids, ts_ids)
