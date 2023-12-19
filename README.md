## Compile without python

Please set the following environment variables using -D option of cmake:

 - Torch_DIR: path to torch(with subpath `/share/cmake/Torch`)
 - LAMMPS_BINARY_ROOT: path to compiled LAMMPS and its library(with `liblammps.so``)
 - LAMMPS_SOURCE_DIR: path to LAMMPS source code(with subpath `/include` to use `lammpsplugin.h`)

**All path should be absolute path.**

And if you use CUDA version of torch, make sure your environment has CUDA environment variables.

You can also set cmake option "-DCMAKE_INSTALL_PREFIX=<path>" to set install target.

### Build Lammps

we need build lammps with shared libs. You can build lammps with cmake as follows:

```
cd lammps/build
cmake -C ../cmake/presets/most.cmake -C ../cmake/presets/oneapi.cmake -D CMAKE_INSTALL_PREFIX=../binary -D BUILD_SHARED_LIBS=yes ../cmake/
```

### Dependencies

- cuda nvcc
- c/c++ complier
- mpi (optional)
- mkl (optional)

We recommend using conda or mamba to install the nvcc/cuda environment, you may need to install using `mamba install cuda=11.8 cuda-nvcc=11.8 cudatoolkit=11.8 -c nvidia`, and make sure using the same version for cudart with libtorch. And also for your pytorch version (11.8 for this example).

And compile lammps with:

(ref to https://github.com/deepmd-kit-recipes/lammps-feedstock/blob/master/recipe/build.sh)

```shell
ARGS="-D PKG_ASPHERE=ON -DPKG_BODY=ON -D PKG_CLASS2=ON -D PKG_COLLOID=ON -D PKG_COMPRESS=OFF -D PKG_CORESHELL=ON -D PKG_DIPOLE=ON -D PKG_EXTRA-COMPUTE=ON -D PKG_EXTRA-DUMP=ON -D PKG_EXTRA-FIX=ON -D PKG_EXTRA-MOLECULE=ON -D PKG_EXTRA-PAIR=ON -D PKG_GRANULAR=ON -D PKG_KSPACE=ON -D PKG_MANYBODY=ON -D PKG_MC=ON -D PKG_MEAM=ON -D PKG_MISC=ON -D PKG_MOLECULE=ON -D PKG_PERI=ON -D PKG_REPLICA=ON -D PKG_RIGID=ON -D PKG_SHOCK=ON -D PKG_SNAP=ON -D PKG_SRD=ON -D PKG_OPT=ON -D PKG_KIM=OFF -D PKG_GPU=OFF -D PKG_KOKKOS=OFF -D PKG_MPIIO=OFF -D PKG_MSCG=OFF -D PKG_LATTE=OFF -D PKG_PHONON=ON -D PKG_REAXFF=ON -D WITH_GZIP=ON -D PKG_COLVARS=ON -D PKG_PLUMED=yes -D PKG_FEP=ON -D PLUMED_MODE=runtime -D PKG_QTB=ON -D PKG_PLUGIN=ON -D PKG_H5MD=ON"
cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_LIB=ON -D BUILD_SHARED_LIBS=ON -D LAMMPS_INSTALL_RPATH=ON -DCMAKE_INSTALL_LIBDIR=lib $ARGS -D FFT=FFTW3 -D CMAKE_INSTALL_PREFIX=【install path】 ../cmake
make #-j${NUM_CPUS}
make install
```
To be briefly, we always need the BUILD_LIB=ON, PKG_PLUGIN=ON, see https://docs.lammps.org/Build_package.html for details from lammps.
Sometimes you may not always need fftw3, so you can delete -D FFT=FFTW3. Or install it following https://docs.lammps.org/Build_settings.html#fft-library.

## make and install

One can make the plugin like:
```bash 
cmake .. -DTorch_DIR=/root/libtorch/libtorch -DLAMMPS_BINARY_ROOT=/root/lammps/binary -DLAMMPS_SOURCE_DIR=/root/lammps -DCMAKE_INSTALL_PREFIX=../binary
```

if you use oneapi, you can add `-C /root/lammps/cmake/presets/oneapi.cmake ` to use same toolchain with lammps.

if you have libtorch in `/root/libtorch/libtorch` and lammps library in `/root/lammps/binary` and lammps source code in `/root/lammps`, and you want to output the plugin in `../binary` as install target.

Or if you use lammps and pytorch from conda, you can set path from conda environment

Thus, you should configure cmake with `cmake -DCMAKE_INSTALL_PREFIX=../binary .. -DTorch_DIR=/opt/libtorch/libtorch/share/cmake/Torch/ -DLAMMPS_BINARY_ROOT=/root/mambaforge/envs/mace/lib/ -DLAMMPS_SOURCE_DIR=/root/lammps-stable_2Aug2023_update1/src`

In this case, libtorch is installed in `/opt/libtorch/libtorch` and lammps binary library is in `/root/mambaforge/envs/mace/lib/` and lammps source code is in `/root/lammps-stable_2Aug2023_update1/src`.

Then you can make and install as:
```
make install -j <cpu_num>
```

## run lammps settings

You can run lammps using plugin but not setting environment variables as follows for test version before it is setting in your environment:
```bash
 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<...>/libtorch/lib:<...>/lammpsPluginTest/binary/lib/ LAMMPS_PLUGIN_PATH=<...>/binary/lib/mace_lmp lmp_mpi -in in.lammps > log
```

- LD_LIBRARY_PATH needs /libtorch/lib of path of `libtorch.so`.
- LAMMPS_PLUGIN_PATH needs the path of `*plugin.so`

## run lammps

just run lammps as follows if you have set the environment variables `LAMMPS_PLUGIN_PATH` correctly:
```bash
lmp_mpi -in in.lammps > log
```

In which the in.lammps consist pair settings as follows:
```
pair_style mace no_domain_decomposition
pair_coeff * * my_mace.model-lammps.pt C H N O
```
Where `my_mace.model-lammps.pt` is the model file by torchscript `python <mace_repo_dir>/mace/cli/create_lammps_model.py my_mace.model`, With no_domain_decomposition, LAMMPS builds a periodic graph rather than treating ghost atoms as independent nodes.

Please notice the pbc neighbor list by lammps is not used in the plugin. So the cutoff by lammps only used for lammps examination. The cutoff in the model file is used for the plugin, not from lammps.