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

def find_keywords_in_sheet(path, sheet_path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            strings = get_shared_strings(z)
            
            with z.open(sheet_path) as f:
                tree = ET.parse(f)
                root = tree.getroot()
                
                targets = ['Total', 'admin0Pcode', 'admin2Name_en']
                print(f"Searching for {targets} in {sheet_path}...")

                for row in root.iter():
                    if row.tag.endswith('row'):
                        row_idx = row.get('r')
                        for cell in row:
                            cell_ref = cell.get('r') # e.g., A1
                            t = cell.get('t')
                            
                            val = None
                            v_el = None
                            for child in cell:
                                if child.tag.endswith('v'):
                                    v_el = child
                                    break
                            
                            if v_el is not None:
                                raw_val = v_el.text
                                if t == 's':
                                    val = strings[int(raw_val)]
                                else:
                                    val = raw_val
                            
                            if val in targets:
                                print(f"Found '{val}' at {cell_ref} (Row {row_idx})")

    except Exception as e:
        print(f"Error: {e}")

find_keywords_in_sheet('data/population/eth_admpop_2023.xlsx', 'xl/worksheets/sheet4.xml')
