import subprocess

subprocess.run([
    '/usr/local/MATLAB/R2024b/bin/matlab',
    '-sd', '/home/mcosta/PycharmProjects/OSCNet/OSCNet',
    '-batch', "addpath('metric'); statistic"
], check=True)
