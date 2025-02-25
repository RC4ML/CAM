#!/bin/bash
export AEOLUS_LOG_LEVEL=ERROR
B=16k
ssd=6
IO=32k
# dd if=/dev/zero of=/share/data6/a bs=1M count=64K
# dd if=/dev/zero of=/share/data6/b bs=1M count=64K
#for N in 32k 48k 64k 80k 96k 112k 128k 160k 192k 224k 256k
# for N in 32768 49152 65536 81920 98304 114688 131072 163840 196608 229376 262144
for N in 32k 48k 64k 80k 96k 112k 128k 160k 192k 224k 256k
do
    for ((i=0; i<1; i++))
    do
        # ./gemm-test-pinned $N $N $N 0 256g 512g $B $IO $ssd
        # ./gemm-cublasxt $N $N $N 0 256g 512g $IO $ssd
        # ./gemm-cublas-gds $N $N $N
        # ~/bam/build.new/bin/nvm-gemm-bench --m=$N --n=$N --k=$N --a_offset=0 --b_offset=274877906944 --c_offset=549755813888 --block_size=16384 --page_size=32768 --blk_size=512 --queue_depth=4096 --pages=524288 --num_queues=128 --threads=4194304 --n_ctrls=$ssd --ssd=1 | grep result
        #./gemm-gds-no-batch $N $N $N /share/data6/a /share/data6/b /share/data6/c $B
        #./gemm-test $N $N $N 0 16UL*1024*1024*1024  32UL*1024*1024*1024  $B $IO 6
        ./gemm-spdk-test $N $N $N 0 16UL*1024*1024*1024  32UL*1024*1024*1024  $B $IO 6
        if [ $? -ne 0 ]; then
            echo "Failed at $N"
            # exit 1
        fi
    done
done
echo "All tests done"