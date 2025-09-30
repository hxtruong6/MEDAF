#!/bin/bash

#PBS -q GPU-1
#PBS -oe
#PBS -lselect=1:ngpus=1
#PBS -N aidan_job
#PBS -m be
#PBS -M hxtruong@jaist.ac.jp

# Customize queue (-q) and job name (-N) as needed

module load cuda/12.8u1
nvidia-smi

# Configuration
HOME_DIR=/home/s2320437                     # CHANGE to your home directory
JUPYTER_PORT=10888                          # Port for Jupyter on supercomputer
JUPYTER_FWD=10888                           # Port to forward to local machine
JUPYTER_CONDA=research_medaf_aidan               # CHANGE to your Conda environment
LOCAL_MACHINE=d053-198.jaist.ac.jp                 # Your local machine IP
LOCAL_SSH_PORT=22                           # CHANGE if SSH port is non-standard
LOCAL_SSH_USER=xuantruong                      # CHANGE to your SSH username
SSH_KEY=~/.ssh/id_ed25519          # CHANGE to your SSH key path

# Activate Conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate $JUPYTER_CONDA
conda --version

# Start Jupyter Lab
echo "Starting Jupyter Lab on port $JUPYTER_PORT..."
jupyter lab --port $JUPYTER_PORT --ip localhost --no-browser &> $PWD/jupyter.log &
JUPYTER_PID=$!
echo "Jupyter PID: $JUPYTER_PID"
sleep 10  # Wait for Jupyter to start

# Display Jupyter URL and token
echo "Jupyter URL and token:"
grep "http://localhost:$JUPYTER_PORT" $PWD/jupyter.log || echo "Failed to find Jupyter URL in log"

# Start reverse SSH tunnel
echo "Starting reverse SSH tunnel to $LOCAL_MACHINE..."
ssh -v -i $SSH_KEY -R 127.0.0.1:$JUPYTER_FWD:localhost:$JUPYTER_PORT -N $LOCAL_SSH_USER@$LOCAL_MACHINE -p $LOCAL_SSH_PORT &
SSH_PID=$!
echo "SSH PID: $SSH_PID"

# Cleanup function
cleanup() {
    echo "Shutting down..."
    [ -n "$JUPYTER_PID" ] && kill $JUPYTER_PID 2>/dev/null
    [ -n "$SSH_PID" ] && kill $SSH_PID 2>/dev/null
    wait
}

trap cleanup SIGINT SIGTERM EXIT

wait
