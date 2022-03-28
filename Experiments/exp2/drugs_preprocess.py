import sys
from rdkit.Chem import MolFromSmiles as Mol

with open(sys.argv[1], 'r') as fr:
    data=fr.read().splitlines()

data = [data[2*i+1] for i in range(int(len(data)/2))]
processed_data = []
for line in data:
    new_line = line[line.index('    ')+4:]
    molecules = new_line.split(', ')
    processed_data += molecules
processed_data = [line+'\n' for line in processed_data]

with open('drugs_2019.smi', 'w') as fw:
    fw.writelines(processed_data)
