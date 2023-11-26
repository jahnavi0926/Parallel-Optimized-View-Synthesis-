#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <netcdf.h>
#include <mpi.h>


#define DT 60.0
#define NX 60
#define NY 60
#define NZ 40
#define DX 1000.0 // Spatial step in meters
#define U0 10.0   // Initial horizontal wind velocity (m/s)
#define V0 5.0    // Initial vertical wind velocity (m/s)
#define Z0 5.0
#define KX 0.00001 // Diffusion coefficient for X-direction
#define KY 0.00001 // Diffusion coefficient for Y-direction
#define KZ 0.00001 // Diffusion coefficient for Z-direction

void initializeField(double field[][NX][NY][NZ], int numSteps) {
    srand(time(NULL));
    for (int t = 0; t < numSteps; t++) {
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for(int k = 0;k<NZ;k++){
                    field[t][i][j][k] = i*j*k;
                }
            }
        }
    }
}

void writeFieldToNetCDF(double field[NX][NY][NZ]) {
    char filename[50];
    sprintf(filename, "output_1_1.nc");

    int ncid;
    int retval;

    if ((retval = nc_create(filename, NC_CLOBBER, &ncid)) != NC_NOERR) {
        fprintf(stderr, "Error creating NetCDF file: %s\n", nc_strerror(retval));
        return;
    }

    int dimids[3];
    int varid;
    int dim_sizes[3] = {NX, NY, NZ};

    if ((retval = nc_def_dim(ncid, "x", NX, &dimids[0])) != NC_NOERR ||
        (retval = nc_def_dim(ncid, "y", NY, &dimids[1])) != NC_NOERR ||
        (retval = nc_def_dim(ncid, "z", NZ, &dimids[2])) != NC_NOERR) {
        fprintf(stderr, "Error defining NetCDF dimensions: %s\n", nc_strerror(retval));
        nc_close(ncid);
        return;
    }

    if ((retval = nc_def_var(ncid, "field", NC_DOUBLE, 3, dimids, &varid)) != NC_NOERR) {
        fprintf(stderr, "Error defining NetCDF variable: %s\n", nc_strerror(retval));
        nc_close(ncid);
        return;
    }

    if ((retval = nc_enddef(ncid)) != NC_NOERR) {
        fprintf(stderr, "Error ending NetCDF definition mode: %s\n", nc_strerror(retval));
        nc_close(ncid);
        return;
    }

    size_t start[3] = {0, 0, 0};
    size_t count[3] = {NX, NY, NZ};

    if ((retval = nc_put_vara_double(ncid, varid, start, count, &field[0][0][0])) != NC_NOERR) {
        fprintf(stderr, "Error writing data to NetCDF file: %s\n", nc_strerror(retval));
        nc_close(ncid);
        return;
    }

    if ((retval = nc_close(ncid)) != NC_NOERR) {
        fprintf(stderr, "Error closing NetCDF file: %s\n", nc_strerror(retval));
        return;
    }

    printf("Saved field data to %s\n", filename);
}

void simulateWeather(double field[][NX][NY][NZ], int rank, int num_processes, int numSteps) {
    double tempField[numSteps][NX][NY][NZ];

    for (int t = 0; t < numSteps; t++) {
        // Advection
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for(int k = 0; k< NZ; k++) {
                    int i_prev = ((int)(i - U0 * DT / DX + NX)) % NX;
                    int j_prev = ((int)(j - V0 * DT / DX + NY)) % NY;
                    int k_prev = ((int)(k - Z0 * DT / DX + NZ)) % NZ;
                    tempField[t][i][j][k] = field[t][i_prev][j_prev][k_prev];
                    // printf("%d %d %d %d : %d %d %d : %lf\n",i,j,k,t,i_prev,j_prev,k_prev,tempField[t][i][j][k]);
                }
                
            }
        }

        // Diffusion
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for(int k = 0;k < NZ; k++){
                    double laplacian = (field[t][(i + 1) % NX][j][k] + field[t][(i - 1 + NX) % NX][j][k]
                                    + field[t][i][(j + 1) % NY][k] + field[t][i][(j - 1 + NY) % NY][k]
                                    + field[t][i][j][(k + 1) % NZ] + field[t][i][j][(k - 1 + NZ) % NZ]
                                    - 6 * field[t][i][j][k]) / (DX * DX);
                tempField[t][i][j][k] += (KX * laplacian + KY * laplacian + KZ*laplacian) * DT;
                // printf("%d %d %d %d : %lf\n",i,j,k,t,tempField[t][i][j][k]);
                }
            }
        }
    }
    
    double l[NX][NY][NZ];
    for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY; j++) {
                for(int k = 0;k < NZ; k++){
                    l[i][j][k] = i*j*k; //tempField[0][i][j][k] + i*j*k;
                }
            }
        }
    

    // Write the field to NetCDF
    if (rank == 0) {
        writeFieldToNetCDF(l);
    }
}

int main(int argc, char **argv) {
    int rank, num_processes;
    int numSteps;

    if (argc != 2) {
        printf("Usage: %s <numSteps>\n", argv[0]);
        return 1;
    }

    numSteps = atoi(argv[1]);
    if (numSteps <= 0) {
        printf("numSteps must be a positive integer.\n");
        return 1;
    }

    double field[numSteps][NX][NY][NZ];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (NX % num_processes != 0) {
        if (rank == 0) {
            fprintf(stderr, "Number of processes must evenly divide NX.\n");
        }
        MPI_Finalize();
        return 1;
    }

    initializeField(field, numSteps); // Initialize the initial field for all time steps

    // Distribute the initial field to all processes
    MPI_Bcast(&field[0][0][0][0], numSteps * NX * NY * NZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    simulateWeather(field, rank, num_processes, numSteps);

    printf("Weather simulation completed.\n");

    MPI_Finalize();
    return 0;
}
