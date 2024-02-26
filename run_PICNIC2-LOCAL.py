import sys, os
import subprocess
# PDB must be in /home/rstewart/resnet/predictions/data-casp15/unpress_TS_model/
assert len(sys.argv) == 2, f'Usage: python {sys.argv[0]} <casp15_pdb_path>\nAn example: python {sys.argv[0]} {sys.argv[0].replace("run_PICNIC2-LOCAL.py","")}local/casp15_af_models/T1104/af2-standard_T1104_1'
assert 'af2-standard' in sys.argv[1], f'Usage: python {sys.argv[0]} <casp15_pdb_path>\nAn example: python {sys.argv[0]} {sys.argv[0].replace("run_PICNIC2-LOCAL.py","")}local/casp15_af_models/T1104/af2-standard_T1104_1'
input_pdb_path = sys.argv[1]
script_dir = os.path.dirname(os.path.abspath(__file__))
model = 'local'

# Generate features
result = subprocess.run(['python', f'{script_dir}/{model}/generate_{model}_feature.py', input_pdb_path], capture_output=True, text=True)

assert result.returncode == 0, result.stderr + result.stdout
features_path = result.stdout.replace('\n','')

# Make predictions
result = subprocess.run(['python', f'{script_dir}/{model}/prediction_{model}.py', features_path, input_pdb_path], capture_output=True, text=True)

assert result.returncode == 0, result.stderr + result.stdout
PDB_f = result.stdout.replace('\n','')
print(f'Output PDB at {PDB_f}')

# Cleanup
os.remove(f'{script_dir}/{model}/.var/df.pkl')
os.remove(f'{script_dir}/{model}/.var/df_esm.pkl')
os.remove(f'{script_dir}/{model}/.var/features.pkl')