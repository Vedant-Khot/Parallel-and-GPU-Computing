Here is the clean, raw content ready to be saved directly into your `README.md` file:

```markdown
# Matrix Multiplication using Sequential, OpenMP, MPI, and CUDA

A comparative parallel computing experiment implementing **4000 × 4000 matrix multiplication** using four different computing models:

- **Sequential C** (Single CPU baseline)
- **OpenMP** (Shared-memory CPU multi-threading)
- **MPI** (Distributed-memory multi-process computing)
- **CUDA** (Massively parallel GPU acceleration)

The project evaluates execution time, speedup, memory overhead, and implementation tradeoffs across each paradigm.

---

## 📌 Problem Definition

Two matrices $A$ and $B$ of size $4000 \times 4000$ are multiplied:

```text
A = 4000 × 4000
B = 4000 × 4000
C = A × B

```

All elements of matrices $A$ and $B$ are initialized to `1.0`.

Therefore:

```text
C[i][j] = Σ A[i][k] × B[k][j]
        = 1.0 + 1.0 + ... + 1.0  (4000 terms)
        = 4000.00

```

The expected verification check for every implementation is:

```text
C[0][0] = 4000.00

```

---

## 📁 Directory Structure

```text
parallel_lab/
├── sequential/
│   └── matrix_sequential.c
├── openmp/
│   └── matrix_openmp.c
├── mpi/
│   ├── matrix_mpi.c
│   └── hosts
├── cuda/
│   └── matrix_cuda.cu
└── README.md

```

---

## ⚙️ Prerequisites & Environment Setup

### 1. Build Tools & OpenMP

```bash
sudo apt update
sudo apt install build-essential -y
gcc --version
nproc

```

### 2. Open MPI

```bash
sudo apt install openmpi-bin libopenmpi-dev -y
mpicc --version
mpirun --version

```

### 3. NVIDIA CUDA Toolkit

```bash
nvidia-smi
nvcc --version

```

---

## 🚀 Compilation and Execution

### 1. Sequential C

Baseline execution on a single CPU core using loop interchange ($i \to k \to j$) for cache-friendly memory access.

```bash
cd ~/parallel_lab/sequential
gcc -O2 matrix_sequential.c -o matrix_sequential
./matrix_sequential

```

**Expected Output:**

```text
Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = 244.120000 seconds
Verification C[0][0] = 4000.00

```

---

### 2. OpenMP (Shared Memory)

Distributes loop iterations across CPU cores sharing a single address space.

```bash
cd ~/parallel_lab/openmp
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
export OMP_NUM_THREADS=8
./matrix_openmp

```

**Expected Output:**

```text
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 8
Execution Time = 30.830434 seconds
Verification C[0][0] = 4000.00

```

---

### 3. MPI (Distributed Memory)

Splits row chunks using `MPI_Scatter`, broadcasts matrix $B$ via `MPI_Bcast`, and gathers computed rows with `MPI_Gather`.

#### Hostfile Setup (`hosts`)

```text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1

```

#### Compile and Run

```bash
cd ~/parallel_lab/mpi
mpicc -O2 matrix_mpi.c -o matrix_mpi

# Run across 4 local processes:
mpirun -np 4 ./matrix_mpi

# Or across a distributed multi-node cluster:
mpirun -np 4 --hostfile hosts ./matrix_mpi

```

**Expected Output:**

```text
MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = 4
Execution Time = 92.979510 seconds
Verification C[0][0] = 4000.00

```

---

### 4. CUDA (GPU Acceleration)

Maps each output matrix cell $C[i][j]$ directly to a 2D grid of CUDA threads.

* **Matrix Size:** $4000 \times 4000$
* **Block Dimensions:** $16 \times 16$ ($256$ threads/block)
* **Grid Dimensions:** $250 \times 250$ ($62,500$ blocks)
* **Total Concurrent Threads:** $16,000,000$

```bash
cd ~/parallel_lab/cuda
nvcc -O2 matrix_cuda.cu -o matrix_cuda
./matrix_cuda

```

**Expected Output:**

```text
CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Grid Size = 250 x 250 blocks
Block Size = 16 x 16 threads
Kernel Execution Time = 0.146443 seconds
Total CUDA Phase Time = 0.165004 seconds
Verification C[0][0] = 4000.00

```

---

## 📊 Benchmark & Speedup Analysis

Speedup is calculated relative to the Sequential CPU baseline:

$$\text{Speedup} = \frac{T_{\text{Sequential}}}{T_{\text{Parallel}}}$$

| Implementation | Model | Resources | Execution Time (s) | Speedup | Verification |
| --- | --- | --- | --- | --- | --- |
| **Sequential** | Single CPU thread | 1 Core | 244.120000 | 1.00× *(Baseline)* | 4000.00 |
| **OpenMP** | Shared memory | 8 CPU Threads | 30.830434 | 7.92× | 4000.00 |
| **MPI** | Distributed memory | 4 Nodes / 4 Processes | 92.979510 | 2.63× | 4000.00 |
| **CUDA** | GPU Acceleration | NVIDIA RTX 4500 Ada | 0.165004 | 1479.48× | 4000.00 |

### Key Observations

1. **OpenMP:** Achieved near-linear scaling ($7.92\times$ on 8 threads) due to minimal thread management overhead and uniform shared-memory access.
2. **MPI:** Suffered from communication overhead (serializing and transmitting matrices across network sockets and VM boundaries), leading to lower efficiency than shared memory.
3. **CUDA:** Achieved massive performance gains ($>1400\times$) by exploiting data-parallel execution across thousands of hardware cores simultaneously.

---

## 🔧 Troubleshooting

| Error / Issue | Cause | Fix |
| --- | --- | --- |
| `fatal error: omp.h: No such file` | Missing compiler flag | Add `-fopenmp` to the GCC compilation command. |
| `mpicc: command not found` | Missing MPI libraries | Run `sudo apt install openmpi-bin libopenmpi-dev`. |
| SSH authentication failed in MPI | Public key missing | Run `ssh-keygen` and propagate with `ssh-copy-id worker_ip`. |
| `nvcc: command not found` | CUDA not in path | Add `export PATH=/usr/local/cuda/bin:$PATH` to `~/.bashrc`. |
| Out of memory error | Insufficient RAM/VRAM | Each matrix requires $\approx 64\text{ MB}$ ($192\text{ MB}$ total). Ensure swap or VRAM is free. |

```

```
