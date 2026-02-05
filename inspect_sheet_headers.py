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

def inspect_sheet_headers(path, sheet_path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            strings = get_shared_strings(z)
            
            with z.open(sheet_path) as f:
                tree = ET.parse(f)
                root = tree.getroot()
                # Find the first row
                # Namespace issue again, use iter
                
                rows = []
                for row in root.iter():
                    if row.tag.endswith('row'):
                        rows.append(row)
                        if len(rows) > 0: break # Just need first row
                
                if not rows:
                    print("No rows found")
                    return

                first_row = rows[0]
                headers = []
                for cell in first_row:
                    # check if shared string
                    t = cell.get('t')
                    v_el = None
                    for child in cell:
                        if child.tag.endswith('v'):
                            v_el = child
                            break
                    
                    if v_el is not None:
                        val = v_el.text
                        if t == 's':
                            headers.append(strings[int(val)])
                        else:
                            headers.append(val)
                    else:
                        headers.append(None)
                        
                print(f"Headers in {sheet_path}:")
                print(headers)

    except Exception as e:
        print(f"Error: {e}")

inspect_sheet_headers('data/population/eth_admpop_2023.xlsx', 'xl/worksheets/sheet4.xml')
