import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from dashboard.google_sheets import (
    get_gsheet_client,
    read_sheet_to_df,
    write_df_to_sheet,
)


def test_get_gsheet_client_returns_gc():
    with patch(
        "dashboard.google_sheets.Credentials.from_service_account_file"
    ) as mock_creds, patch("dashboard.google_sheets.gspread.authorize") as mock_auth:
        mock_auth.return_value = MagicMock()
        gc = get_gsheet_client("dummy.json")
        assert gc is not None


def test_read_sheet_to_df_returns_dataframe():
    gc = MagicMock()
    sh = gc.open_by_key.return_value
    worksheet = sh.worksheet.return_value
    worksheet.get_all_records.return_value = [{"a": 1, "b": 2}]
    df = read_sheet_to_df(gc, "spreadsheet", "sheet")
    assert isinstance(df, pd.DataFrame)
    assert "a" in df.columns


def test_write_df_to_sheet_writes():
    gc = MagicMock()
    sh = gc.open_by_key.return_value
    worksheet = sh.worksheet.return_value
    worksheet.clear = MagicMock()
    worksheet.update = MagicMock()
    df = pd.DataFrame({"a": [1], "b": [2]})
    write_df_to_sheet(gc, "spreadsheet", "sheet", df)
    worksheet.clear.assert_called()
    worksheet.update.assert_called()
