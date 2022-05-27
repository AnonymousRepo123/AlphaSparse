# spmv_builder

所有的操作：
1、对角块分解
2、密集块分解
3、稠密行分块
4、稠密列分块
5、排序
6、压缩
7、压缩行分块
8、压缩列分块
9、索引压缩
10、padding

CUDA纠错：
https://cloud.tencent.com/developer/news/374520

退火算法：
https://www.cnblogs.com/heaad/archive/2010/12/20/1911614.html

计时：
https://zhuanlan.zhihu.com/p/54665895

模板，用递归的方式嵌套不同层次的循环，BLB reduce的碎片里面包含WLB的碎片，WLB里面的碎片包含TLB的碎片。

nvprof --metrics achieved_occupancy,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,gld_transactions,gst_transactions,gld_transactions_per_request,gst_transactions_per_request

#### Description
A build for SpMV

#### Software Architecture
Software architecture description

#### Installation

1.  xxxx
2.  xxxx
3.  xxxx

#### Instructions

1.  xxxx
2.  xxxx
3.  xxxx

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)

https://blog.csdn.net/weixin_44621617/article/details/107818700

git log --pretty=tformat: --numstat | gawk '{ add += $1 ; subs += $2 ; loc += $1 - $2 } END { printf "added lines: %s removed lines : %s total lines: %s\n",add,subs,loc }' -

安装其他版本的gcc在家目录：
https://blog.csdn.net/qq_33278461/article/details/106357783

安装其他版本的nvcc在家目录：
https://www.cxyzjd.com/article/qq_35599200/108937533

C++的虚函数（父指针调用子实例的方法）：
https://blog.csdn.net/ly890700/article/details/55803398
https://segmentfault.com/a/1190000023597934
https://blog.csdn.net/xukang95/article/details/106397411

Turing架构一些指令和IO的周期数量：
常量内存的几级索引，L1、L1.5、L2对于一个Cache line的访问延迟分别是，26、100、200个周期
对于shared mem来说，在没有bank conflict的前提下，读延迟大概20个周期
在TLB不溢出的前提下，直接对于全局内存的访问大概300个周期，如果TLB没有被cache到，那大概600个周期
L1 cache的延迟大概32个周期，L2 190个周期。
原子加，在没有冲突或者少量冲突的前提下，在全局内存中大概平均75个周期，在shared mem中大概10个周期

merge-based的核心逻辑：
cuda_code/merge-spmv/cub/agent/agent_spmv_orig.cuh 500-600行。
merge-based的归约逻辑和CSR5的主要区别，线程内归约，CSR5通过bitmap来看换行（bitmap一开始从全局内存读到寄存器中，一个线程一个，之后从寄存器中一位一位读），merge通过当前nz index（通过二分搜索获得）与row offset的比较来看换行（row offset一开始从全局内存读到共享内存中，一个线程块一组）。
线程间归约，都是使用segment sum的策略，CSR5用warp reduce，但是segment sum的关键参数做减法的便宜是从全局内存读的。merge-based用shared memory，因为每个线程第一个中间结果的行索引是线程推测出来的，所以不需要进一步的IO。

写作：
定语从句：https://www.cpsenglish.com/question/28317

卸载NV驱动：
https://chujian521.github.io/blog/2021/01/05/NVIDIA驱动安装之禁用nouveau/

在Ubuntu20中安装gcc-4
https://blog.csdn.net/CharlieVV/article/details/111242143#:~:text=Step%201%20-%20%E5%AE%89%E8%A3%85%20%E9%BB%98%E8%AE%A4%20%E7%89%88%E6%9C%ACgcc%20%EF%BC%9A%20,GCC%E7%89%88%E6%9C%AC%20%E6%98%AF%20GCC%204.4.1%EF%BC%8C%E6%89%80%E4%BB%A5%E9%9C%80%E8%A6%81%E8%87%AA%E5%B7%B1%20%E5%AE%89%E8%A3%85%E4%BD%8E%E7%89%88%E6%9C%AC%20%E7%9A%84%20GCC%204.1.2%E3%80%8
https://www.cnblogs.com/zzoo/p/ubuntu20_04_3-gcc_4_8_5.html
https://chrisjean.com/fix-apt-get-update-the-following-signatures-couldnt-be-verified-because-the-public-key-is-not-available/

版本切换，环境变量名称放到最后：
https://blog.csdn.net/FX677588/article/details/78681325

形式化系统：
https://www.lumin.tech/articles/lambda-calculus/
PFPL：https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi_m56FmM32AhULL6YKHWehCPYQFnoECA8QAQ&url=http%3A%2F%2Fwww.cs.cmu.edu%2F~rwh%2Fpfpl.html&usg=AOvVaw0MowzEdcRLg4cNlNg2dKdE
yu zhang：http://staff.ustc.edu.cn/~yuzhang/ 及其合作者
符号表：https://en.wikipedia.org/wiki/List_of_logic_symbols


后台运行：
2835744
nohup ./main > data_source/test.log 2>&1 &
27851
nohup python3 test_from_UF_dataset.py 3Dspectralwave > data_source/test.log 2>&1 &

grep main | xargs ps -u duzhen

srun -w cn15,cn16,cn17 sleep 1000&