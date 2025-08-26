```sh
qsub -q GPU-S -I

module load cuda/12.1

nvcc --version # check CUDA version
nvidia-smi # check GPU
```