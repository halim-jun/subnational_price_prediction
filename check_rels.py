import zipfile
import xml.etree.ElementTree as ET
import os

def list_rels(path):
    try:
        with zipfile.ZipFile(path, 'r') as z:
            with z.open('xl/_rels/workbook.xml.rels') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                # Find Relationship for rId4
                for child in root:
                    if child.get('Id') == 'rId4':
                        print(f"Target for rId4: {child.get('Target')}")
                        return child.get('Target')
    except Exception as e:
        print(f"Error: {e}")

list_rels('data/population/eth_admpop_2023.xlsx')
