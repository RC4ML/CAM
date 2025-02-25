#!/bin/bash
export AEOLUS_LOG_LEVEL=ERROR
B=16k
ssd=6
IO=32k

for N in 32k 48k 64k 80k 96k 112k 128k 160k 192k 224k 256k
do
    for ((i=0; i<3; i++))
    do
        .build/application/gemm/gemm-test $N $N $N 0 16UL*1024*1024*1024  32UL*1024*1024*1024  $B $IO 6
        if [ $? -ne 0 ]; then
            echo "Failed at $N"
            # exit 1
        fi
    done
done
echo "All tests done"