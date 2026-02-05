import zipfile
import xml.etree.ElementTree as ET

def get_shared_strings(z):
    with z.open('xl/sharedStrings.xml') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        names = []
        for t in root.iter():
            if t.tag.endswith('t'):
                names.append(t.text)
    return names

def inspect_all_sheets_headers(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            strings = get_shared_strings(z)
            
            # Identify all sheet xmls
            sheet_files = [f for f in z.namelist() if f.startswith('xl/worksheets/sheet')]
            sheet_files.sort()
            
            for sheet_path in sheet_files:
                print(f"\n--- {sheet_path} ---")
                with z.open(sheet_path) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    
                    rows = []
                    for row in root.iter():
                        if row.tag.endswith('row'):
                            rows.append(row)
                            if len(rows) >= 2: break 
                    
                    if not rows:
                        print("Empty")
                        continue

                    # Print first 2 rows
                    for i, row in enumerate(rows):
                        row_vals = []
                        for cell in row:
                            t = cell.get('t')
                            v_el = None
                            for child in cell:
                                if child.tag.endswith('v'):
                                    v_el = child
                                    break
                            
                            val = None
                            if v_el is not None:
                                raw_val = v_el.text
                                if t == 's':
                                    val = strings[int(raw_val)]
                                else:
                                    val = raw_val
                            row_vals.append(val)
                        print(f"Row {i+1}: {row_vals}")

    except Exception as e:
        print(f"Error: {e}")

inspect_all_sheets_headers('data/population/eth_admpop_2023.xlsx')
