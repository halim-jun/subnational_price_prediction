import zipfile
import xml.etree.ElementTree as ET

def inspect_xlsx_headers(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            with z.open('xl/sharedStrings.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                strings = [t.text for t in root.iter() if t.tag.endswith('t')]
                
                # Heuristic: Headers are often distinct words, often in the beginning, or matching keywords
                keywords = ['Admin', 'Region', 'Zone', 'Woreda', 'Pop', 'Total', 'Male', 'Female', 'Name', 'Code']
                print("Potential Headers found:")
                for s in strings:
                    if s and any(k.lower() in s.lower() for k in keywords):
                        print(s)
                
                print("\nFirst 20 strings:")
                print(strings[:20])

    except Exception as e:
        print(f"Error: {e}")

inspect_xlsx_headers('data/population/eth_admpop_2023.xlsx')
