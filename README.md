# CAM: Asynchronous GPU-Initiated, CPU-Managed SSD Management for Batching Storage Access


This is the source code for our paper.



## Required hardware and software

- NVIDIA 80GB-PCIe-A100
- HugePage: At least 4096 huge pages
- g++ >= 11.3.0



## Install Dependencies and Build
See [INSTALL.md](./doc/INSTALL.md) for install dependencies and build RPCNIC on a single machine.


## Run Test
If Check if the configuration is correct in Run Experiments of [EXP.md](./doc/EXP.md) passes, then everything will be fine. Please refer to exp.md for more details.


### Directory Structure:

~~~
.
├── spdk (our modified SPDK driver)
├── gdrcopy (our modified gdrcopy driver)
├── build.sh (Script for compiling CAM code project)
├── doc （Document on how to install and conduct experiments with CAM）
├── README.md
└── src
    ├── benchmarks
    │   ├── CAM_benchmark （microbenchmark for CAM,one thread control one SSD）
    │   │ 
    │   └── CAM_variable_core_benchmark (microbenchmark for CAM,one thread control variable SSDs)
    │       
    ├── CAM_lib (source code of CAM)
    │   
    ├── CAM_variable_core_lib (source code of CAM with one thread control variable SSDs)
    │  
    └── GPU_memory_lib (source code for GPU memory management used in CAM)
    │  
    └── applications
        └── gemm (end to end test in GEMM application)
       
~~~





### Getting help

Working in the process...



### Contact

email at songziyu@zju.edu.cn


