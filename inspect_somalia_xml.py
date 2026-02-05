import zipfile
import xml.etree.ElementTree as ET

def inspect_somalia_region_xml(path):
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
                
                print("Row | Col 2 (Region?) | Col 3 (District?) | Col 4 (P_Code?)")
                count = 0
                for row in root.iter():
                    if row.tag.endswith('row'):
                        row_num = row.get('r')
                        vals = {} # map col letter/idx to value
                        
                        for cell in row:
                            r = cell.get('r') # e.g., C2
                            # Simple parsing of column
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
                            
                            vals[col] = val
                        
                        # Assuming Col 2 is C, Col 3 is D, Col 4 is E?
                        # Wait, A=1, B=2, C=3.
                        # My previous dump showed:
                        # Row 2: ['1', 'Total', 'Banadir', 'Bondhere'...]
                        # Col 0 (A) = '1'
                        # Col 1 (B) = 'Total'
                        # Col 2 (C) = 'Banadir'
                        # Col 3 (D) = 'Bondhere'
                        
                        c_val = vals.get('C', '')
                        d_val = vals.get('D', '')
                        e_val = vals.get('E', '')
                        
                        if count < 20:
                            print(f"{row_num} | {c_val} | {d_val} | {e_val}")
                        count += 1

    except Exception as e:
        print(f"Error: {e}")

inspect_somalia_region_xml('data/population/somalia_population.xlsx')
