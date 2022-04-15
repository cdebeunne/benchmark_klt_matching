import os
import subprocess
import pandas as pd

def run_sequence(run_file, param_file_name, result_file):
    # do this stupid chdir twice before main
    # is not in the same place as the python script    
    os.chdir('build')
    subprocess.run([run_file, param_file_name])
    os.chdir('..')
    df = pd.read_csv(result_file)
    return df
