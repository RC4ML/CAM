# Installation

This document shows all of the essential software installation process on test machine. 

## 1. Install gdrcopy Driver:


~~~bash
cd gdrcopy
make
make install
~~~


## 2. Config Hugepages

Our system need enough Hugepages, before run application, first alloc hugepages.

~~~bash
sudo sh -c "echo 32768 > /proc/sys/vm/nr_hugepages"
~~~


## 3. Build SPDK:

~~~bash
cd spdk
git submodule update --init
./configure --with-shared
make
~~~

## 4. Install SPDK Driver on NVMe SSDs.
~~~bash
cd spdk/scrips
sudo ./setup.sh
~~~

## 5. Build all CAM software

Now we create a `buildt` directory, and build all the software in the `build` directory.
~~~bash
mkdir build
bash build.sh
~~~

It should report no error. And we will get the output binary in the `build` directory.


## 4. Uninstall SPDK driver on NVMe SSD.

After experiment, you need to uninstall SPDK driver
~~~bash
cd spdk/scrips
sudo ./setup.sh reset
~~~

## 5. Extra Attention

Before installing the SPDK driver, please ensure that there is no data present on the SSD.

During the installation of the SPDK driver on NVMe SSDs, it is possible that some SSDs may not be able to install the SPDK driver. This is mainly due to the SSD already being mounted or having a file system on it. Therefore, it is necessary to unmount the SSD and wipe the file system before installing the SPDK driver.

To unmount and wipe the file system on the selected SSDs, use the following command:
~~~bash
sudo umount /dev/nvmeXn1
sudo wipefs -a /dev/nvmeXn1
~~~