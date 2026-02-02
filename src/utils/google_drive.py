import pandas as pd
import re
import requests
from io import BytesIO

def get_drive_id(url):
    """Extracts the file ID from a Google Drive URL."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    # Also handle 'id=' format if passed
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def read_drive_file(url, **kwargs):
    """
    Reads a Google Drive file directly into a customized pandas DataFrame.
    Tries to detect if it's CSV or Excel.
    
    Args:
        url (str): Google Drive file URL.
        **kwargs: Arguments passed to pd.read_csv or pd.read_excel.
    """
    file_id = get_drive_id(url)
    if not file_id:
        raise ValueError("Could not extract file ID from URL")
        
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # Check if it's a direct downloadeable link or needs confirmation (large files)
    # For simple public files, we can just pass the URL to pandas often, 
    # but using requests + BytesIO is more robust for "without downloading to disk"
    
    response = requests.get(download_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch file: Status {response.status_code}")
        
    # Try reading as CSV
    try:
        return pd.read_csv(BytesIO(response.content), **kwargs)
    except Exception as e_csv:
        # Try reading as Excel
        try:
            return pd.read_excel(BytesIO(response.content), **kwargs)
        except Exception as e_excel:
            raise ValueError(f"Could not read file as CSV or Excel. \nCSV Error: {e_csv}\nExcel Error: {e_excel}")
