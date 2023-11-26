# CS677-Parallel-optimised-View-Synthesis

## Setup for our code

1. Create a conda environment using our environment.yml file.
   ```
   conda env create --name envname --file=environments.yml 

   conda activate envname 
   ```
2. Create viewpoints using make_data.py with N_VIEWS viewpoints
   ```
   python3 make_data.py N_VIEWS
   ```

4. Run the MPI code surf_optim for surface rendering for NUM_PROCESSES processes
   ```
    mpiexec -n NUM_PROCESSES pvbatch surf_optim.py
   ```
   Run the code vol_optim for volume rendering for NUM_PROCESSES processes
   ```
    mpiexec -n NUM_PROCESSES pvbatch vol_optim.py
   ```

   The initial images will be saved in InitImages and Optimised images will be saved in OptiImages
   
6. The time taken for surface rendering optimization is stored in results1.csv and volume rendering optimization is stored in results2.csv. The columns in results1/2.csv represent number of processes, number of viewpoints, time taken.
   
7. This cam be analysed by opttime.
   
8. Similarity results can be reproduced using similaritygraphs.ipynb. The similarity and opttime codes are in the helper files.
   
