#!/bin/bash
#SBATCH --partition raphael
#SBATCH --mincpus 16
#SBATCH --mem 164G
#SBATCH --time 7-0:00:00
#SBATCH --job-name jupyter
#SBATCH --output jupyter-notebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i10001-20000 -n1)
node=$(hostname -s)
user=$(whoami)
cluster="ros"

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.cs.princeton.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.cs.princeton.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
# load modules or conda environments here
source /n/fs/ragr-data/users/viola/mouse_dev/.venv/bin/activate
cd /n/fs/ragr-data/users/viola/structuredVAE/

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
/n/fs/ragr-data/users/viola/mouse_dev/.venv/bin/jupyter-lab --no-browser --port=${port} --ip=${node}
