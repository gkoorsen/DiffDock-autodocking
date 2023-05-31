from google.colab import drive 
import pandas as pd
from google.colab import files
import os
import requests
import time
from random import random
from tqdm import tqdm 
from glob import glob
from shlex import quote
from datetime import datetime
from tqdm.auto import tqdm
import re
import datetime
import subprocess
import numpy as np

def install_all_packages(): 


  if os.path.exists('/content/DiffDock/results'):
    !rm -rf /content/DiffDock/results

  !pip install ipython-autotime --quiet
  %load_ext autotime

  if not os.path.exists('/content/DiffDock'):
      %cd /content
      !git clone https://github.com/gcorso/DiffDock.git
      %cd /content/DiffDock
      !git checkout a6c5275 # remove/update for more up to date code

  try:
      import biopandas
      import torch
  except:
      !pip install pyg==0.7.1 --quiet
      !pip install pyyaml==6.0 --quiet
      !pip install scipy==1.7.3 --quiet
      !pip install networkx==2.6.3 --quiet
      !pip install biopython==1.79 --quiet
      !pip install rdkit-pypi==2022.03.5 --quiet
      !pip install e3nn==0.5.0 --quiet
      !pip install spyrmsd==0.5.2 --quiet
      !pip install biopandas==0.4.1 --quiet
      !pip install torch==1.12.1+cu113 --quiet

  import torch

  try:
      import torch_geometric
  except ModuleNotFoundError:
      !pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
      !pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html --quiet
      !pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html --quiet
      !pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html --quiet
      !pip install git+https://github.com/pyg-team/pytorch_geometric.git  --quiet 

  if not os.path.exists("/content/DiffDock/esm"):
      %cd /content/DiffDock
      !git clone https://github.com/facebookresearch/esm
      %cd /content/DiffDock/esm
      !git checkout ca8a710 # remove/update for more up to date code
      !sudo pip install -e .
      %cd /content/DiffDock

def import_data_and_create_csv(file_name,output_file):
  df = pd.read_excel(file_name)
  PDBs = ''
  for name in list(df['PDB IDs']):  
    PDBs += name+',' 
  PDB_id = PDBs[:-1]

  def download_pdb_file(pdb_id: str) -> str:
      """Download pdb file as a string from rcsb.org"""
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
              url = f"https://alphafold.ebi.ac.uk/files/{pdb_id}-model_v3.pdb"
          else:
              url = f"http://files.rcsb.org/view/{pdb_id}.pdb"
          filename = f'{pdb_id}.pdb'

      cache_path = os.path.join(PDB_DIR, filename)
      if os.path.exists(cache_path):
          return cache_path

      pdb_req = requests.get(url)
      pdb_req.raise_for_status()
      open(cache_path, 'w').write(pdb_req.text)
      return cache_path

  pdb_files = [download_pdb_file(_PDB_id) for _PDB_id in tqdm(PDB_id.split(","))]
  smiless = df['Compound SMILES']

  with open(output_file, 'w') as out:
      out.write("protein_path,ligand\n")
      for pdb_file in pdb_files:
        for smiles in smiless:
          out.write(f"{pdb_file},{smiles}\n")

def import_data(file_name,output_file):

  df = pd.read_excel(file_name)
  PDBs = ''
  for name in list(df['PDB IDs']):  
    PDBs += name+',' 
  PDB_id = PDBs[:-1]

  def download_pdb_file(pdb_id: str) -> str:
      """Download pdb file as a string from rcsb.org"""
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
              url = f"https://alphafold.ebi.ac.uk/files/{pdb_id}-model_v3.pdb"
          else:
              url = f"http://files.rcsb.org/view/{pdb_id}.pdb"
          filename = f'{pdb_id}.pdb'

      cache_path = os.path.join(PDB_DIR, filename)
      if os.path.exists(cache_path):
          return cache_path

      pdb_req = requests.get(url)
      pdb_req.raise_for_status()
      open(cache_path, 'w').write(pdb_req.text)
      return cache_path

  pdb_files = [download_pdb_file(_PDB_id) for _PDB_id in tqdm(PDB_id.split(","))]
  smiless = df['Compound SMILES'].dropna()

  return pdb_files, smiless

def run_in_batches_per_pdb(pdb_files,smiless,batch_length):

  df_list = []
  i = 0
  length = len(pdb_files)
  done = 0
  remainder = length - done

  while remainder > 0:
    i += 1
    if remainder >= batch_length:
      with open('/tmp/input_protein_ligand.csv', 'w') as out:
        out.write("protein_path,ligand\n")
        for pdb_file in pdb_files[done:done+batch_length]:
          for smiles in smiless:
            out.write(f"{pdb_file},{smiles}\n")
      done += batch_length
      print(f'{round(done/length*100,2)} % done...')

    else:
      with open('/tmp/input_protein_ligand.csv', 'w') as out:
        out.write("protein_path,ligand\n")
        for pdb_file in pdb_files[done:]:
          for smiles in smiless:
            out.write(f"{pdb_file},{smiles}\n")
      done += batch_length

    remainder = length - done

    %cd /content/DiffDock
    !python datasets/esm_embedding_preparation.py --protein_ligand_csv /tmp/input_protein_ligand.csv --out_file data/prepared_for_esm.fasta 
    %cd /content/DiffDock
    %env HOME=esm/model_weights
    %env PYTHONPATH=$PYTHONPATH:/content/DiffDock/esm
    !python /content/DiffDock/esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm.fasta data/esm2_output --repr_layers 33 --include per_tok --truncation_seq_length 30000
    %cd /content/DiffDock
    !python -m inference --protein_ligand_csv /tmp/input_protein_ligand.csv --out_dir results/user_predictions_small --inference_steps 20 --samples_per_complex 10 --batch_size 6
    %cd /content/DiffDock/results/user_predictions_small

    confidences = np.load('/content/DiffDock/results/user_predictions_small/confidences.npy')
    names = np.load('/content/DiffDock/results/user_predictions_small/complex_names.npy')

    complexes = [(name[name.find('-tmp-pdb-')+10:name.find('.pdb_')],name[name.find('.pdb____')+8:]) for name in names]

    df_results = pd.DataFrame({'PDB' : [pdb for pdb,smiles in complexes],
                            'smiles' : [smiles for pdb,smiles in complexes],
                            'top confidence score' : [max(c) for c in confidences],
                            'mean confidence score' : [np.mean(c) for c in confidences],
                            'min confidence score' : [min(c) for c in confidences]})

    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d_%H-%M-%S')

    df_results_tsv = f"df_diffdock_results_batch_{i}_{now_str}.tsv"
    df_results.to_csv(f'/content/gdrive/MyDrive/DiffDock results/Anathi/{df_results_tsv}', sep='\t', index=None)
    df_list.append(df_results)

  combined_results = pd.DataFrame()
  results_name = f"combined_results_{datetime.now().isoformat()[2:10].replace('-','')}"
  for d in df_list:
    combined_results = pd.concat([combined_results,d])
  combined_results.to_csv(f'/content/gdrive/MyDrive/DiffDock results/Anathi/{results_name}_', sep='\t', index=None)

def warnings_check(process):

  try:
    stdout = process.stdout.decode('utf-8')
  except:
    stdout = ''
  try:
    stderr = process.stderr.decode('utf-8')
  except:
    stderr = ''

  # Look for warnings in the stderr text
  warnings = [line for line in stderr.split('\n') if 'warning' in line.lower()]
  warnings.extend([line for line in stdout.split('\n') if 'warning' in line.lower()])
  
  for warning in warnings:
      print(warning)

def run_in_batches_per_pdb_python(pdb_files,smiless,batch_length):

  df_list = []
  i = 0
  length = len(pdb_files)
  done = 0
  remainder = length - done

  while remainder > 0:

    for _ in tqdm(range(remainder), desc='Processing batches', unit='batch'):
      i += 1
      if remainder >= batch_length:
        with open('/tmp/input_protein_ligand.csv', 'w') as out:
          out.write("protein_path,ligand\n")
          for pdb_file in pdb_files[done:done+batch_length]:
            for smiles in smiless:
              out.write(f"{pdb_file},{smiles}\n")
        done += batch_length

      else:
        with open('/tmp/input_protein_ligand.csv', 'w') as out:
          out.write("protein_path,ligand\n")
          for pdb_file in pdb_files[done:]:
            for smiles in smiless:
              out.write(f"{pdb_file},{smiles}\n")
        done += batch_length

      remainder = length - done

    # Define the directory paths
      base_dir = "/content/DiffDock"
      esm_dir = os.path.join(base_dir, "esm")
      results_dir = os.path.join(base_dir, "results/user_predictions_small")

      # Run the esm_embedding_preparation.py script
      os.chdir(base_dir)
      print('Preparing ESM embeddings...')
      ESM = subprocess.run(["python", "datasets/esm_embedding_preparation.py", "--protein_ligand_csv", "/tmp/input_protein_ligand.csv", "--out_file", "data/prepared_for_esm.fasta"], check=True)
      warnings_check(ESM)


      # Set up environment variables for ESM
      os.environ['HOME'] = os.path.join(esm_dir, "model_weights")
      os.environ['PYTHONPATH'] += ":" + esm_dir

      # Run the extract.py script
      os.chdir(base_dir)
      print('Running extract.py...')
      extract = subprocess.run(["python", os.path.join(esm_dir, "scripts/extract.py"), "esm2_t33_650M_UR50D", "data/prepared_for_esm.fasta", "data/esm2_output", "--repr_layers", "33", "--include", "per_tok", "--truncation_seq_length", "30000"], check=True)
      warnings_check(extract)

      # Run the inference module
      os.chdir(base_dir)
      print('Performing DiffDock inference...')
      inference = subprocess.run(["python", "-m", "inference", "--protein_ligand_csv", "/tmp/input_protein_ligand.csv", "--out_dir", "results/user_predictions_small", "--inference_steps", "20", "--samples_per_complex", "10", "--batch_size", "6"], check=True)
      warnings_check(inference)

      # Change to the results directory
      os.chdir(results_dir)
      confidences = np.load('/content/DiffDock/results/user_predictions_small/confidences.npy')
      names = np.load('/content/DiffDock/results/user_predictions_small/complex_names.npy')

      complexes = [(name[name.find('-tmp-pdb-')+10:name.find('.pdb_')],name[name.find('.pdb____')+8:]) for name in names]

      df_results = pd.DataFrame({'PDB' : [pdb for pdb,smiles in complexes],
                              'smiles' : [smiles for pdb,smiles in complexes],
                              'top confidence score' : [max(c) for c in confidences],
                              'mean confidence score' : [np.mean(c) for c in confidences],
                              'min confidence score' : [min(c) for c in confidences]})

      now = datetime.datetime.now()
      now_str = now.strftime('%Y-%m-%d_%H-%M-%S')

      df_results_tsv = f"df_diffdock_results_batch_{i}_{now_str}.tsv"
      df_results.to_csv(f'/content/gdrive/MyDrive/DiffDock results/Anathi/{df_results_tsv}', sep='\t', index=None)
      df_list.append(df_results)

  combined_results = pd.DataFrame()

  now = datetime.datetime.now()
  now_str = now.strftime('%Y-%m-%d_%H-%M-%S')

  results_name = f"combined_results_{now_str}.csv"
  
  for d in df_list:
    combined_results = pd.concat([combined_results,d])
  combined_results.to_csv(f'/content/gdrive/MyDrive/DiffDock results/Anathi/{results_name}', sep='\t', index=None)

  return combined_results

  
def main():

  drive.mount('/content/gdrive')
  install_all_packages()
  results = run_in_batches_per_pdb_python(pdb_files, smiless,batch_length=1)


if __name__ == '__main__':
  main()

