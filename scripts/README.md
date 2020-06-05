#### Example submission of BBTRE simulation
```bash
sbatch execute-MITgcm-mpi-repeat rotated_BBTRE 0 720000
```

#### Programming environments

We use two separate Anaconda distributions since the two were found to be incompatible.

For notebooks that use the `dedalus` package, we use the `bblturb` environment, which is created with the commands
```bash
bash install_conda.sh
bash update_conda.sh
```

For analyzing MITgcm output, we use the `bblturb-analysis` environment.