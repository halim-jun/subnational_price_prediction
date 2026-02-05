import zipfile
import xml.etree.ElementTree as ET

def inspect_somalia_patterns(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            # Load strings
            strings = []
            try:
                with z.open('xl/sharedStrings.xml') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    for t in root.iter():
                        if t.tag.endswith('t'):
                            strings.append(t.text)
            except:
                pass
            
            with z.open('xl/worksheets/sheet1.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                
                print(f"{'Row':<4} | {'Col 2 (Region)':<20} | {'Col 3 (District)':<20} | {'Col 4 (P_Code)':<10}")
                print("-" * 60)
                
                rows_to_show = []
                for row in root.iter():
                    if row.tag.endswith('row'):
                        row_vals = {}
                        row_num = int(row.get('r'))
                        
                        for cell in row:
                            r = cell.get('r')
                            col = "".join([c for c in r if c.isalpha()])
                            t = cell.get('t')
                            
                            val = ""
                            v_el = None
                            for child in cell:
                                if child.tag.endswith('v'):
                                    v_el = child
                                    break
                            
                            if v_el is not None:
                                raw_val = v_el.text
                                if t == 's':
                                    if strings:
                                        val = strings[int(raw_val)]
                                    else:
                                        val = raw_val
                                else:
                                    val = raw_val
                            row_vals[col] = val
                        
                        # Map C->Region, D->District, E->P_Code
                        c_val = row_vals.get('C', '')
                        d_val = row_vals.get('D', '')
                        e_val = row_vals.get('E', '')
                        
                        # Store first 25 rows
                        if row_num <= 25:
                            print(f"{row_num:<4} | {c_val:<20} | {d_val:<20} | {e_val:<10}")

    except Exception as e:
        print(f"Error: {e}")

inspect_somalia_patterns('data/population/somalia_population.xlsx')
