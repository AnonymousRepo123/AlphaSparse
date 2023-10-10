Please go to https://github.com/duzhen1996/AlphaSparse for the latest version.

# AlphaSparse

AlphaSparse is a superset of traditional auto-tuners that goes beyond the scope of existing human-designed format(s) and implementation(s). It automatically creates novel machine-designed formats and SpMV kernel implementations entirely from the knowledge of input sparsity patterns and hardware architectures.

The SC'22 paper: https://dl.acm.org/doi/abs/10.5555/3571885.3571972

## Hardware Configurations

- We have tested on NVIDIA RTX 2080 and A100. Generally, AlphaSparse can support all up-to-date NVIDIA GPUs.
- CUDA version: 11.x. The nvcc needs to be configured into environment variables.
- GCC version: 9.4 or any others supporting C++11.
- Operating System version: Ubuntu 20.04.3 LTS, Linux kernel 5.4.0-99-generic x86_64.
- Python version: 3.x.
- No third-part library is needed.

## Download data of test matrices

In our evaluation, all input matrices are from SuiteSparse Matrix Collection (https://sparse.tamu.edu). Two ways are provided to download data as follows.

- Directly download from websites. Go to the websites of SuiteSparse Matrix Collection, click the link of specific matrix name, and click the download link named "Matrix Market". The downloaded file is zipped. By extracting the file, a ".mtx" file can be gotten.

- Use the python interface. Install the `ssgetpy` Python module. Run `import ssgetpy` and type `help(ssgetpy)` to get a detailed help message on using ssgetpy to search and download sparse matrices. We have provided a python script named `get_data_set_from_UF.py` to download all the needed matrices. The variable `UF_DIR` is the destination of downloaded data. The script needs `wget`.

## Run AlphaSparse

Go to the root directory of AlphaSparse, and compile the source code.

```
make clean
make -j16
```

Create a directory named `data_source`.

```
mkdir data_source
```

Configure AlphaSparse according to the environment. Go to `global_config.json`. Set two fields named `ROOT_PATH_STR` and `spmv_header_file` need to be set according to the path of AlphaSparse.

```
"ROOT_PATH_STR": "{path to AlphaSparse}"
"spmv_header_file": "{path to AlphaSparse}/spmv_header_top.code"
```

Prepare the input matrix. Go to the directory of UF dataset. Extract the zipped file of specific matrix. And convert the `.mtx` file.

```
cd {path of UF dateset}
tar -zxvf {matrix name}.tar.gz
python3 data_prepare.py {path of UF dateset}/{matrix name}/{matrix name}.mtx data_source/{matrix name}.mtx.coo
```

Run AlphaSparse. The description of the best Operator Graph and the corresponding performance are shown in test.log.

```
./main data_source/{matrix name}.mtx.coo >> data_source/test.log
```

## Batch Test of AlphaSparse

We have provided a script named `batch_test_spmv_builder.py` to test all matrices in our paper. All matrix names are stored in matrix_name_list. 

```
python3 batch_test_spmv_builder.py >> data_source/test.log
```

## Batch Test of its counterparts

Run `./make.sh` in `cuda_code/ACSR_test`, `cuda_code/CSR_adptive_test`, `cuda_code/ELL_test`, `cuda_code/taco-csr`. Run `make` in `cuda_code/CSR5_cuda`. Run `./coo_make.sh` and `./csr_make.sh`. 

Test cuSparse.

```
python3 batch_test_cusparse_from_UF.py
```

Test CSR5, ELL, ACSR, CSR-Adaptive.

```
python3 get_data_set_from_UF.py
```

Change CUDA version to 9.2. Run `make gpu_spmv` in `cuda_code/merge-spmv`. Run batch test of Merge and HYB.

```
python3 batch_test_merge_hyb_from_UF.py
```
