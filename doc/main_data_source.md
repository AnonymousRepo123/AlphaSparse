# 一些在已有的文章中有意思的数据集

**在CSR adptive使用的矩阵**

（CSR5做得比较好的矩阵，在ACSR中做得更好）

https://sparse.tamu.edu/Mittelmann/rail4284


（几个比较均衡的矩阵）

https://sparse.tamu.edu/Williams/pdb1HYS （在yaSpMV中做得很好的矩阵）

https://sparse.tamu.edu/Bates/sls


（超级不均衡的几个矩阵）

https://sparse.tamu.edu/IBM_EDA/dc2 （这个矩阵CSR5也有很好的表现）

https://sparse.tamu.edu/LAW/eu-2005 （这个矩阵CSR5也有很好的表现）

https://sparse.tamu.edu/Freescale/transient


**在CSR5这篇文章中认为的有代表性的矩阵，CSR5在均衡的矩阵中比较一般**


CSR5开源：https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/tree/master/CSR5_cuda


（非常均衡的矩阵）

https://sparse.tamu.edu/Williams/cant

https://sparse.tamu.edu/Williams/mac_econ_fwd500 （在均衡矩阵中CSR5做得不错的）

https://sparse.tamu.edu/Hamm/scircuit （在均衡矩阵中CSR5做得不错的）


（很不均衡的矩阵中，CSR5做得非常好的）

https://sparse.tamu.edu/Williams/webbase-1M （高度压缩的归约元数据，以及幂律分布）



**在ACSR中很牛逼的矩阵（Fast Sparse Matrix-Vector Multiplication on GPUs for Graph Applications）**

源码：https://github.com/aneesh297/Sparse-Matrix-Vector-Multiplication/blob/master/ACSR_Implementation/ACSR_new.cu

https://sparse.tamu.edu/Mittelmann/rail4284 （前文提及的ACSR中做得比较好）

https://sparse.tamu.edu/LAW/in-2004

https://sparse.tamu.edu/LAW/indochina-2004 （一个很大的矩阵，一条一条的稠密和稀疏条带交错分布，稠密的部分和稀疏的部分的边界很清楚的矩阵）



**一些比较没有代表性的，但是在已有的文章中一直出现的**


https://sparse.tamu.edu/Williams/consph （FEM/Spheres）

https://sparse.tamu.edu/Williams/mc2depi （mc2depi）

https://sparse.tamu.edu/PARSEC/Ga41As41H72 （Ga41As41H72）

https://sparse.tamu.edu/PARSEC/Si41Ge41H72 （Si41Ge41H72）

https://sparse.tamu.edu/Freescale/circuit5M （circuit5M，这个太大了，怕爆内存）

https://sparse.tamu.edu/Oberwolfach/bone010

https://sparse.tamu.edu/QCD/conf6_0-8x8-30

https://sparse.tamu.edu/Rucci/Rucci1

https://sparse.tamu.edu/Boeing/pwtk

https://sparse.tamu.edu/GHS_indef/boyd2

https://sparse.tamu.edu/Andrianov/ins2





# 中等规模的其他数据集

数据集大概是10^6个非零元，用ell padding rate来代表行非零元数量的差异程度。

ell padding rate的计算是由（ell格式padding之后的大小 / 原本的大小）


padding rate:

rajat29 60150

Stanford_Berkeley 7520

flickr 857

web-NotreDame 749

tx2010 24

Hardesty2 1.5


为了保证内存不爆炸，只在tx2010和Hardesty2中执行ELL格式的spmv

格式分别是coo、ell、sell、csr5





