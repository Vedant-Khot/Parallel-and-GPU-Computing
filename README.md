Below is an updated `README.md` based on the lab manual you provided. It includes **Sequential, OpenMP, MPI, CUDA, compilation commands, execution instructions, results, speedup, troubleshooting, and project structure**.

 Save the following content as `README.md`.

````
# Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

A comparative parallel computing experiment implementing **4000 × 4000 matrix multiplication** using four different computing models:

- Sequential C
- OpenMP shared-memory parallelism
- MPI distributed-memory parallelism
- CUDA GPU parallelism

The project compares execution time, speedup, resource usage, and the practical differences between sequential CPU execution, shared-memory parallelism, distributed-memory parallelism, and GPU acceleration.

---

## 📌 Problem Definition

Two `4000 × 4000` matrices are multiplied:

```text
A = 4000 × 4000
B = 4000 × 4000
C = A × B
````

 All elements of matrices `A` and `B` are initialized to `1.0`.

 Therefore:

```
C[i][j] = Σ A[i][k] × B[k][j]

         = 1 + 1 + 1 + ... + 1
           └──── 4000 terms ────┘

         = 4000
```

 The expected verification result is:

```
C[0][0] = 4000.00
```

 The same mathematical operation and verification value are maintained across all four implementations.

---

 # 🧩 Implementations

 ## 1\. Sequential C

 The sequential implementation performs matrix multiplication using a single CPU execution flow.

```
Matrix A + Matrix B
        ↓
    Single CPU
        ↓
     Matrix C
```

 ### Source File

```
matrix_sequential.c
```

 ### Compile

```
gcc -O2 matrix_sequential.c -o matrix_sequential
```

 ### Run

```
./matrix_sequential
```

 ### Expected Output

```
Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = ...
Verification C[0][0] = 4000.00
```

 ### Reference Result

```
Execution Time = 244.120000 seconds
Verification C[0][0] = 4000.00
```

 The sequential implementation is used as the baseline for calculating speedup.

---

 # 2\. OpenMP Shared-Memory Parallelism

 OpenMP parallelizes the outer matrix multiplication loop and distributes rows among multiple CPU threads.

```
                 Matrix A
                    +
                 Matrix B
                    ↓
          ┌───────────────────┐
          │   Shared Memory   │
          └───────────────────┘
             ↓    ↓    ↓    ↓
            T1   T2   T3   ... T8
             ↓    ↓    ↓    ↓
                Matrix C
```

 The main parallel region uses:

```
#pragma omp parallel for private(j, k)
```

 ### Source File

```
matrix_openmp.c
```

 ### OpenMP Requirements

 Install the GCC development tools:

```
sudo apt update
sudo apt install build-essential -y
```

 Verify GCC:

```
gcc --version
```

 Verify the number of available CPU cores:

```
nproc
```

 Set the number of OpenMP threads:

```
export OMP_NUM_THREADS=8
```

 Verify:

```
echo $OMP_NUM_THREADS
```

 ### Compile

 OpenMP requires the `-fopenmp` compiler option:

```
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

 ### Run

```
./matrix_openmp
```

 ### Expected Output

```
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 8
Execution Time = ...
Verification C[0][0] = 4000.00
```

 ### Reference Result

```
Execution Time = 30.830434 seconds
Threads = 8
Verification C[0][0] = 4000.00
```

---

 # 3\. MPI Distributed-Memory Parallelism

 MPI distributes the matrix multiplication workload across multiple processes running on separate virtual machines.

 The reference configuration uses:

```
4 MPI processes
4 virtual machines
1000 rows per MPI process
```

 The data flow is:

```
                     Matrix A
                        |
                 MPI_Scatter
                        |
        +---------------+---------------+
        |               |               |
      Rank 0          Rank 1          Rank 2       Rank 3
    1000 rows       1000 rows       1000 rows    1000 rows
        |               |               |             |
        +---------------+---------------+-------------+
                        |
                    Computation
                        |
                    local_C
                        |
                 MPI_Gather
                        |
                Complete Matrix C
                   on Rank 0
```

 Matrix `B` is distributed to every process using:

```
MPI_Bcast()
```

 Matrix `A` is divided among processes using:

```
MPI_Scatter()
```

 The partial results are collected using:

```
MPI_Gather()
```

---

 ## MPI Requirements

 Install Open MPI:

```
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev -y
```

 Verify MPI:

```
mpicc --version
```

 and:

```
mpirun --version
```

---

 ## MPI Source

```
matrix_mpi.c
```

 Compile:

```
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

---

 ## MPI Host Configuration

 For the four-VM configuration, create a hostfile:

```
hosts
```

 Example:

```
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

 The actual hostnames and IP addresses should match the configured virtual machines.

---

 ## Test SSH Connectivity

 From the Master VM:

```
ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname
```

 For passwordless SSH, configure SSH keys:

```
ssh-keygen
```

 Then copy the public key to each worker:

```
ssh-copy-id worker1
ssh-copy-id worker2
ssh-copy-id worker3
```

 Test again:

```
ssh worker1 hostname
```

---

 ## Run MPI

 From the Master VM:

```
mpirun -np 4 --hostfile hosts ./matrix_mpi
```

 If the executable is stored in the user's home directory:

```
mpirun -np 4 --hostfile hosts sh -c '$HOME/matrix_mpi'
```

 The executable must be available on the participating worker machines according to the MPI environment being used.

 ### Expected Result

```
MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = 4
Execution Time = ...
Verification C[0][0] = 4000.00
```

 ### Reference Result

```
Execution Time = 92.979510 seconds
MPI Processes = 4
Verification C[0][0] = 4000.00
```

---

 # 4\. CUDA GPU Parallelism

 CUDA offloads the matrix multiplication to an NVIDIA GPU.

 The CPU prepares the matrices and transfers them to GPU memory. A CUDA kernel is then launched, where each logical CUDA thread computes one output matrix element.

```
             CPU
              |
        Matrix Initialization
              |
        Host → Device
              |
              ↓
       ┌───────────────┐
       │   NVIDIA GPU  │
       │               │
       │ CUDA Threads  │
       │      ↓        │
       │ Matrix C      │
       └───────────────┘
              |
        Device → Host
              |
              ↓
        Verification
```

---

 ## CUDA Requirements

 The CUDA implementation requires:

 - NVIDIA CUDA-capable GPU
- NVIDIA driver
- CUDA Toolkit
- `nvcc` compiler
- Supported host compiler
- Sufficient GPU memory

---

 ## Verify NVIDIA GPU

 Run:

```
nvidia-smi
```

 The reference system uses:

```
NVIDIA RTX 4500 Ada Generation
```

 The actual GPU available on your system may be different.

---

 ## Verify CUDA Compiler

 Run:

```
nvcc --version
```

 This confirms that the CUDA Toolkit and `nvcc` compiler are available.

---

 ## CUDA Project Directory

 Create the CUDA directory:

```
mkdir -p ~/parallel_lab/cuda
cd ~/parallel_lab/cuda
```

---

 ## CUDA Source File

 The CUDA source file is:

```
matrix_cuda.cu
```

 Create it using:

```
nano matrix_cuda.cu
```

 The CUDA implementation uses:

```
__global__ void matMulKernel(float *A, float *B, float *C, int n)
```

 Each CUDA thread determines its output position using:

```
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

---

 ## CUDA Compilation

 Compile using:

```
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

 No compilation error should be reported.

---

 ## Run CUDA Program

```
./matrix_cuda
```

 ### Expected Output

```
CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Grid Size = 250 x 250 blocks
Block Size = 16 x 16 threads
Kernel Execution Time = ...
Total CUDA Phase Time = ...
Verification C[0][0] = 4000.00
```

---

 # ⚙️ CUDA Execution Configuration

 The reference configuration uses:

 | Parameter | Configuration |
| --- | --- |
| Matrix Size | 4000 × 4000 |
| Block Size | 16 × 16 |
| Threads per Block | 256 |
| Grid Size | 250 × 250 |
| Total Blocks | 62,500 |
| Logical CUDA Threads | 16,000,000 |

Since:

```
4000 / 16 = 250
```

 the CUDA grid contains:

```
250 × 250 = 62,500 blocks
```

 Each block contains:

```
16 × 16 = 256 threads
```

 Therefore:

```
62,500 × 256 = 16,000,000
```

 logical thread instances.

 The logical threads correspond to the output elements of the `4000 × 4000` matrix, with boundary checks protecting against out-of-range threads.

---

 # 📊 Results and Performance Comparison

 The following values are the reference measurements from the lab manual.

 All implementations produced:

```
C[0][0] = 4000.00
```

 | Implementation | Model | Resources | Execution Time | Verification |
| --- | --- | --- | --- | --- |
| Sequential | Single CPU execution | 1 CPU core | 244.120000 s | 4000.00 |
| OpenMP | Shared memory | 8 CPU threads | 30.830434 s | 4000.00 |
| MPI | Distributed memory | 4 processes / 4 VMs | 92.979510 s | 4000.00 |
| CUDA | GPU parallelism | NVIDIA RTX 4500 Ada | 0.165004 s | 4000.00 |

---

 # 🚀 Speedup Comparison

 Speedup is calculated relative to the sequential baseline:

```
Speedup = Sequential Execution Time / Parallel Execution Time
```

 Using the reference measurements:

 | Implementation | Execution Time | Speedup |
| --- | --- | --- |
| Sequential | 244.120000 s | 1.00× |
| OpenMP | 30.830434 s | 7.92× |
| MPI | 92.979510 s | 2.63× |
| CUDA | 0.165004 s | 1479.48× |

These values are measurements from the reference lab environment. Actual execution times and speedups will vary depending on CPU, GPU, memory, virtualization, network configuration, compiler versions, and system load.

---

 # 📈 CUDA Timing

 The CUDA implementation reports two different measurements.

 ### Kernel-only time

 Reference:

```
0.146443 seconds
```

 This measures the GPU kernel execution.

 ### Total CUDA phase time

 Reference:

```
0.165004 seconds
```

 The total CUDA phase includes:

```
Host → Device transfer
        +
GPU Kernel execution
        +
Device → Host transfer
```

 Therefore, the total CUDA phase time is the more appropriate value when comparing the complete CUDA computation phase with the other implementations.

---

 # 🔍 Performance Observations

 ## Sequential

 The sequential implementation performs the complete matrix multiplication using one CPU execution flow.

 It provides the baseline execution time against which the parallel implementations can be compared.

---

 ## OpenMP

 OpenMP distributes iterations of the outer matrix multiplication loop among CPU threads.

 With eight threads, the reference execution time was:

```
30.830434 seconds
```

 The main advantage is that all threads operate within the same shared-memory address space.

---

 ## MPI

 MPI distributes the workload between separate processes and, in the reference setup, across four virtual machines.

 The communication flow includes:

```
MPI_Scatter
MPI_Bcast
MPI_Gather
```

 Communication and network/virtualization overhead are part of the distributed-memory execution model.

 Reference execution time:

```
92.979510 seconds
```

---

 ## CUDA

 CUDA maps the output matrix elements to GPU threads.

 For the reference configuration:

```
Matrix = 4000 × 4000
Block = 16 × 16
Grid = 250 × 250
```

 The reference total CUDA phase time was:

```
0.165004 seconds
```

 The measured value includes memory transfers as well as GPU execution.

---

 # 📁 Suggested Project Structure

```
parallel_lab/
│
├── sequential/
│   ├── matrix_sequential.c
│   └── matrix_sequential
│
├── openmp/
│   ├── matrix_openmp.c
│   └── matrix_openmp
│
├── mpi/
│   ├── matrix_mpi.c
│   ├── matrix_mpi
│   └── hosts
│
├── cuda/
│   ├── matrix_cuda.cu
│   └── matrix_cuda
│
└── README.md
```

---

 # 🛠️ Compilation Summary

 ## Sequential

```
gcc -O2 matrix_sequential.c -o matrix_sequential
```

 Run:

```
./matrix_sequential
```

---

 ## OpenMP

```
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

 Set threads:

```
export OMP_NUM_THREADS=8
```

 Run:

```
./matrix_openmp
```

---

 ## MPI

```
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

 Run:

```
mpirun -np 4 --hostfile hosts ./matrix_mpi
```

---

 ## CUDA

```
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

 Run:

```
./matrix_cuda
```

---

 # 🔧 Troubleshooting

 | Problem | Solution |
| --- | --- |
| `gcc: command not found` | Run `sudo apt update && sudo apt install build-essential -y` |
| `stdio.h: No such file or directory` | Install/reinstall the development libraries with `sudo apt install --reinstall gcc libc6-dev` |
| OpenMP compilation fails | Make sure the compile command contains `-fopenmp` |
| OpenMP uses fewer threads | Check `nproc` and `echo $OMP_NUM_THREADS` |
| `mpicc: command not found` | Install `openmpi-bin` and `libopenmpi-dev` |
| MPI SSH connection fails | Check VM networking, hostnames/IP addresses and SSH configuration |
| SSH asks for a password | Configure SSH keys using `ssh-keygen` and `ssh-copy-id` |
| `mpirun` cannot launch workers | Check `hosts`, SSH access and executable availability |
| `nvidia-smi` fails | Check NVIDIA driver installation and GPU visibility |
| `nvcc: command not found` | Install/configure the CUDA Toolkit and PATH |
| CUDA compilation fails | Verify CUDA Toolkit version and supported host compiler |
| Memory allocation fails | Reduce `N` or ensure sufficient system/GPU memory |

---

 # 🧪 Recommended Verification

 ## Sequential

 Run:

```
gcc --version
```

 Then:

```
./matrix_sequential
```

 Verify:

```
C[0][0] = 4000.00
```

---

 ## OpenMP

 Run:

```
nproc
echo $OMP_NUM_THREADS
```

 Then:

```
./matrix_openmp
```

 Verify:

```
Number of Threads Used = 8
C[0][0] = 4000.00
```

---

 ## MPI

 Test connectivity:

```
ssh worker1 hostname
ssh worker2 hostname
ssh worker3 hostname
```

 Then:

```
mpirun -np 4 --hostfile hosts ./matrix_mpi
```

 Verify:

```
Number of MPI Processes = 4
C[0][0] = 4000.00
```

---

 ## CUDA

 Check GPU:

```
nvidia-smi
```

 Check compiler:

```
nvcc --version
```

 Compile:

```
nvcc -O2 matrix_cuda.cu -o matrix_cuda
```

 Run:

```
./matrix_cuda
```

 Verify:

```
C[0][0] = 4000.00
```

---

 # 📸 Recommended Screenshots for Lab Submission

 ## Sequential

 Capture:

 1. `gcc --version`
2. Source code
3. Compilation command
4. Program execution
5. Final output
6. `C[0][0] = 4000.00`

---

 ## OpenMP

 Capture:

 1. `nproc`
2. `echo $OMP_NUM_THREADS`
3. OpenMP source code
4. Compilation command
5. `htop` showing CPU threads
6. Final program output

---

 ## MPI

 Capture:

 1. VM network configuration
2. Worker connectivity
3. SSH test
4. `hosts` file
5. MPI compilation
6. `mpirun` execution
7. MPI rank output
8. Final verification

---

 ## CUDA

 Capture:

 1. `nvidia-smi`
2. `nvcc --version`
3. CUDA source code
4. Compilation command
5. CUDA execution
6. Grid and block configuration
7. Kernel execution time
8. Total CUDA phase time
9. Final verification

---

 # 📝 Conclusion

 This experiment implements the same `4000 × 4000` matrix multiplication using four different computing models:

```
Sequential CPU
      ↓
OpenMP Shared Memory
      ↓
MPI Distributed Memory
      ↓
CUDA GPU Parallelism
```

 The sequential implementation establishes the CPU baseline.

 OpenMP demonstrates shared-memory CPU parallelism by distributing loop iterations among multiple threads.

 MPI demonstrates distributed-memory parallelism by distributing matrix data and computation across multiple processes and virtual machines.

 CUDA demonstrates GPU-based parallelism by assigning matrix output elements to GPU threads.

 All four implementations use the same mathematical workload and produce the same verification result:

```
C[0][0] = 4000.00
```

 The reference measurements demonstrate how execution characteristics change across sequential, shared-memory, distributed-memory, and GPU-based implementations. Actual performance will depend on the hardware, software versions, number of CPU threads, VM configuration, network performance, GPU model, and system load.

---

 # 📚 Quick Command Reference

```
# Sequential
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential

# OpenMP
export OMP_NUM_THREADS=8
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
./matrix_openmp

# MPI
mpicc -O2 matrix_mpi.c -o matrix_mpi
mpirun -np 4 --hostfile hosts ./matrix_mpi

# CUDA
nvcc -O2 matrix_cuda.cu -o matrix_cuda
./matrix_cuda
```

 Expected verification for every implementation:

```
C[0][0] = 4000.00
```

```

```
#   P a r a l l e l - a n d - G P U - C o m p u t i n g  
 