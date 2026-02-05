import zipfile
import xml.etree.ElementTree as ET

def get_shared_strings(z):
    try:
        with z.open('xl/sharedStrings.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            names = []
            for t in root.iter():
                if t.tag.endswith('t'):
                    names.append(t.text)
            return names
    except:
        return []

def inspect_somalia_rows(path, sheet_path='xl/worksheets/sheet1.xml'):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            strings = get_shared_strings(z)
            
            with z.open(sheet_path) as f:
                tree = ET.parse(f)
                root = tree.getroot()
                
                rows_data = []
                for row in root.iter():
                    if row.tag.endswith('row'):
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
                                    if strings:
                                        val = strings[int(raw_val)]
                                    else:
                                        val = f"StringRef:{raw_val}"
                                else:
                                    val = raw_val
                            row_vals.append(val)
                        rows_data.append(row_vals)
                        if len(rows_data) >= 5: break 
                
                print(f"First 5 rows in {sheet_path}:")
                for i, r in enumerate(rows_data):
                    print(f"Row {i+1}: {r}")

    except Exception as e:
        print(f"Error: {e}")

inspect_somalia_rows('data/population/somalia_population.xlsx')
