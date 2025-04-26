import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# --- Google Sheets Helper Functions ---
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_gsheet_client(json_keyfile_path: str):
    """
    Authenticate using a service account JSON keyfile and return a gspread client.
    """

    creds = Credentials.from_service_account_file(json_keyfile_path, scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc


def read_sheet_to_df(gc, spreadsheet_id: str, worksheet_name: str) -> pd.DataFrame:
    """
    Read a worksheet from Google Sheets and return as a pandas DataFrame.
    Raises an exception if the sheet or worksheet is not found.
    """

    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.worksheet(worksheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data)


def write_df_to_sheet(
    gc, spreadsheet_id: str, worksheet_name: str, df: pd.DataFrame
) -> None:
    """
    Write a pandas DataFrame to a Google Sheets worksheet, replacing its contents.
    Raises an exception if the sheet or worksheet is not found.
    """

    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.worksheet(worksheet_name)
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
