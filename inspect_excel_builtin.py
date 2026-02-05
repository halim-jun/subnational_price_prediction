import zipfile
import xml.etree.ElementTree as ET

def inspect_xlsx(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            # Load shared strings
            with z.open('xl/sharedStrings.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                # Namespace usually: http://schemas.openxmlformats.org/spreadsheetml/2006/main
                # Find all 't' text elements
                strings = [t.text for t in root.iter() if t.tag.endswith('t')]
                print("First 50 strings found:")
                print(strings[:50])
                
    except Exception as e:
        print(f"Error: {e}")

inspect_xlsx('data/population/eth_admpop_2023.xlsx')
