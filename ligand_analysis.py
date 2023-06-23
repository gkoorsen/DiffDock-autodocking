from google.colab import files
import pandas as pd
import os
import requests


def download_pdb_file(pdb_id: str) -> str:

    PDB_DIR ="/tmp/pdb/"
    os.makedirs(PDB_DIR, exist_ok=True)

    # url or pdb_id
    if pdb_id.startswith('http'):
        url = pdb_id
        filename = url.split('/')[-1]
    elif pdb_id.endswith(".pdb"):
        return pdb_id
    else:
        if pdb_id.startswith("AF"):
            url = f'https://alphafold.ebi.ac.uk/files/{pdb_id}-model_v3.pdb'
        else:
            url = f'http://files.rcsb.org/view/{pdb_id}.pdb'
        filename = f'{pdb_id}.pdb'

    cache_path = os.path.join(PDB_DIR, filename)
    if os.path.exists(cache_path):
        return cache_path

    pdb_req = requests.get(url)
    pdb_req.raise_for_status()
    open(cache_path, 'w').write(pdb_req.text)
    return cache_path

def find_ligands(pdb_file):

  ligand_counts = {}
  ligand_names = {}

  with open(pdb_file, 'r') as f:
    for line in f:
        if line.startswith("HETNAM"):
            ligand_code = line.split()[1].strip()
            ligand_name = line.split()[2].strip()
            ligand_names[ligand_code] = ligand_name
        elif line.startswith("HET "):
            het_ligand = line.split()[1].strip()
            chain = line.split()[2].strip()[0]
            if het_ligand not in ligand_counts:
                ligand_counts[het_ligand] = {'chain' : [], 'counts' : []}
            if chain not in ligand_counts[het_ligand]['chain']:
                ligand_counts[het_ligand]['chain'].append(chain)
                ligand_counts[het_ligand]['counts'].append(1)
            else:
                position = ligand_counts[het_ligand]['chain'].index(chain)
                ligand_counts[het_ligand]['counts'][position] += 1

  return ligand_names, ligand_counts


def analyse_ligands(pdb_files):

  data = pd.DataFrame()
  pdbs = []
  ligs = []
  names = []
  chains = []
  counts = []

  for pdb_file in pdb_files:
      ligand_dict, ligand_count_dict = find_ligands(pdb_file)
      ligands = list(ligand_count_dict.keys())

      # Display ligands with counts
      for ligand in ligands:
        for i,chain in enumerate(ligand_count_dict[ligand]['chain']):
          pdbs.append(pdb_file[pdb_file.find('/pdb/')+5:pdb_file.find('.pdb')])
          ligs.append(ligand)
          names.append(ligand_dict[ligand])
          chains.append(chain)
          counts.append(ligand_count_dict[ligand]['counts'][i])

  data['PDB'] = pdbs
  data['Ligands'] = ligs
  data['Names'] = names
  data['Chain'] = chains
  data['Count'] = counts

  return data


def main():
  uploaded = files.upload()
  file_name = list(uploaded.keys())[0]
  df = pd.read_excel(file_name)

  PDB_files = []

  for pdb_id in df['PDBs'].dropna():
    PDB_files.append(download_pdb_file(pdb_id.strip()))

  data = analyse_ligands(PDB_files)
  data.to_excel(f'{file_name}_ligand_count.xlsx', index = None)
  files.download(f'{file_name}_ligand_count.xlsx')

if __name__ == '__main__':
  main()
