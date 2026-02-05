import zipfile
import xml.etree.ElementTree as ET

def find_string_globally(path, target):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            # Check shared strings first
            with z.open('xl/sharedStrings.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                found_in_strings = False
                for i, t in enumerate(root.iter()):
                    if t.tag.endswith('t') and t.text == target:
                        print(f"'{target}' found in shared strings at index {i}")
                        found_in_strings = True
                        break
                
                if not found_in_strings:
                    print(f"'{target}' NOT found in shared strings.")
                    return

            # If found in strings, find usages in sheets
            sheet_files = [f for f in z.namelist() if f.startswith('xl/worksheets/sheet')]
            for sheet in sheet_files:
                with z.open(sheet) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    found_in_sheet = False
                    # We can't easily check values without mapping strings, but we know it exists.
                    # This check is just to confirm existence in the file at all.
                    pass # logic requires full mapping which is slow.
            
    except Exception as e:
        print(f"Error: {e}")

find_string_globally('data/population/eth_admpop_2023.xlsx', 'admin0Pcode')
