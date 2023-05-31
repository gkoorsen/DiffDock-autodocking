
import pandas as pd
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
import subprocess
import numpy as np


def import_data(file_name):

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
    
    os.chdir('~/DiffDock')
    os.mkdir('tmp')

    for _ in tqdm(range(remainder), desc='Processing batches', unit='batch'):
      i += 1
      if remainder >= batch_length:
        with open('~DiffDock/tmp/input_protein_ligand.csv', 'w') as out:
          out.write("protein_path,ligand\n")
          for pdb_file in pdb_files[done:done+batch_length]:
            for smiles in smiless:
              out.write(f"{pdb_file},{smiles}\n")
        done += batch_length

      else:
        with open('~/DiffDock/tmp/input_protein_ligand.csv', 'w') as out:
          out.write("protein_path,ligand\n")
          for pdb_file in pdb_files[done:]:
            for smiles in smiless:
              out.write(f"{pdb_file},{smiles}\n")
        done += batch_length

      remainder = length - done

    # Define the directory paths
      base_dir = "~/DiffDock"
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
      try:
        inference = subprocess.run(["python", "-m", "inference", "--protein_ligand_csv", "/tmp/input_protein_ligand.csv", "--out_dir", "results/user_predictions_small", "--inference_steps", "20", "--samples_per_complex", "10", "--batch_size", "6"], check=True)
        warnings_check(inference)
        # Change to the results directory
        os.chdir(results_dir)
        confidences = np.load('~/DiffDock/results/user_predictions_small/confidences.npy')
        names = np.load('~/DiffDock/results/user_predictions_small/complex_names.npy')

        complexes = [(name[name.find('-tmp-pdb-')+10:name.find('.pdb_')],name[name.find('.pdb____')+8:]) for name in names]

        df_results = pd.DataFrame({'PDB' : [pdb for pdb,smiles in complexes],
                                'smiles' : [smiles for pdb,smiles in complexes],
                                'top confidence score' : [max(c) for c in confidences],
                                'mean confidence score' : [np.mean(c) for c in confidences],
                                'min confidence score' : [min(c) for c in confidences]})

        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d_%H-%M-%S')

        df_results_tsv = f"df_diffdock_results_batch_{i}_{now_str}.tsv"
        df_results.to_csv(f'~/DiffDock/results/{df_results_tsv}', sep='\t', index=None)
        df_list.append(df_results)
      except:
        try:
          stderr = process.stderr.decode('utf-8')
          print(stderr)
        except:
          print('Undefined error occurred...')

  combined_results = pd.DataFrame()

  now = datetime.datetime.now()
  now_str = now.strftime('%Y-%m-%d_%H-%M-%S')

  results_name = f"positive_results_{now_str}.csv"
  
  for d in df_list:
    combined_results = pd.concat([combined_results,d])

  combined_results_filtered = combined_results[combined_results['top confidence score'] > 0]
  combined_results_filtered.to_csv(f'~/DiffDock/results/{results_name}', sep='\t', index=None)

  return combined_results

  
def main(input_file):
  
  pdb_files, smiless =  import_data(input_file)
  results = run_in_batches_per_pdb_python(pdb_files, smiless,batch_length=2)


if __name__ == '__main__':
  main(input_file)

