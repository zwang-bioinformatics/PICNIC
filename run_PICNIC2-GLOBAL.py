import sys, os
import subprocess
assert len(sys.argv) == 2, f'Usage: python {sys.argv[0]} <input_pdb_path>'
input_pdb_path = sys.argv[1]
script_dir = os.path.dirname(os.path.abspath(__file__))
model = 'global'

# Generate features
result = subprocess.run(['python', f'{script_dir}/{model}/generate_{model}_feature.py', input_pdb_path], capture_output=True, text=True)

assert result.returncode == 0, result.stderr
features_path = result.stdout.replace('\n','')

# Make predictions
result = subprocess.run(['python', f'{script_dir}/{model}/prediction_{model}.py', features_path, input_pdb_path], capture_output=True, text=True)

assert result.returncode == 0, result.stderr
PDB_f = result.stdout.replace('\n','')
print(f'Output PDB at {PDB_f}')

# Cleanup
os.remove(f'{script_dir}/{model}/.var/df.pkl')
os.remove(f'{script_dir}/{model}/.var/df_esm.pkl')
os.remove(f'{script_dir}/{model}/.var/features.pkl')