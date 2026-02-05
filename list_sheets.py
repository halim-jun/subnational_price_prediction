import zipfile
import xml.etree.ElementTree as ET

def list_sheets(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            with z.open('xl/workbook.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                # Namespace usually: http://schemas.openxmlformats.org/spreadsheetml/2006/main
                # Find 'sheets' element then child 'sheet' elements
                # The tag name might include namespace, so we search by local name
                sheets = []
                for sheet in root.iter():
                    if sheet.tag.endswith('sheet'):
                        name = sheet.get('name')
                        sheet_id = sheet.get('sheetId')
                        r_id = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id') # r:id
                        sheets.append((name, sheet_id, r_id))
                
                print("Sheets found:")
                for s in sheets:
                    print(f"Name: {s[0]}, ID: {s[1]}, r:id: {s[2]}")
                    
    except Exception as e:
        print(f"Error: {e}")

list_sheets('data/population/eth_admpop_2023.xlsx')
