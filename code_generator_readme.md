# 代码生成器的设计
一个代码生成器，代码生成器首先需要数据的依赖分析，找出在各个层次所需要的变量。这些变量出现在每个层次的一开始遍历的一开始和代码的一开始。首先需要一个类来规范来规范所有的变量以及变量的初始化操作。声明的位置一般在循环的外层，第一次初始化的位置在循环的内层:
```
kernel_function()
{
    global metadata define
    global metadata set
    
    thread block level metadata define
    for (BLB traverse with thread block level sync)
    {
        thread block level metadata set (with thread block level sync)
        
        warp level metadata define
        for (WLB traverse)
        {
            warp level metadata set
            
            thread level metadata define
            for (TLB traverse)
            {
                thread level metadata set
                
                multiply non-zero and element in vector and thread level reduce
            }

            warp level reduce
        }

        block level reduce 
    }

    global level reduce, maybe it isn't needed.
}
```