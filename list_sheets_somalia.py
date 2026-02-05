import zipfile
import xml.etree.ElementTree as ET

def list_sheets_somalia(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            with z.open('xl/workbook.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                sheets = []
                for sheet in root.iter():
                    if sheet.tag.endswith('sheet'):
                        name = sheet.get('name')
                        sheet_id = sheet.get('sheetId')
                        r_id = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id') # r:id
                        sheets.append((name, sheet_id, r_id))
                
                print("Sheets found in Somalia file:")
                for s in sheets:
                    print(f"Name: {s[0]}, ID: {s[1]}, r:id: {s[2]}")
                    
    except Exception as e:
        print(f"Error: {e}")

list_sheets_somalia('data/population/somalia_population.xlsx')
