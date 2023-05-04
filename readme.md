# Installation

## mlpack

First, install the dependencies:

1. `sudo apt-get install liblapack-dev`
2. `sudo apt-get install libblas-dev`
3. `sudo apt-get install libboost-dev`
4. `sudo apt-get install libarmadillo-dev`
5. `sudo apt-get install libensmallen-dev`
6. `sudo apt-get install libcereal-dev`

Then, `mlpack` should be built and installed from according to the GitHub page: https://github.com/mlpack/mlpack

1. `git clone https://github.com/mlpack/mlpack`
2. `cd mlpack`
3. `mkdir build && cd build/`
4. `cmake ..`
5. `sudo make install`

## conan

1. [Install Conan](https://docs.conan.io/2/installation.html)
2. Generate a Conan profile by running `conan profile detect --force`
3. The generated profile (located in `~/.conan2/profiles`) should resemble
   ```
   [settings]
   arch=x86_64
   build_type=Release
   compiler=gcc
   compiler.cppstd=gnu14
   compiler.libcxx=libstdc++11
   compiler.version=11
   os=Linux
   ```
4. `cd` into working directory
5. `conan install . -pr:b=default --output-folder=build --build=missing`
6. `cd build`
7. `cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release ..`
8. `make`

## Event Sensor Detection and Tracking

1. Clone the Event Sensor Detection and Tracking repo into the working directory
    * `cd .../LiveTracking`
    * `gh repo clone I2SL/Event-Sensor-Detection-and-Tracking`

## FLIR PTU-SDK
The FLIR PTU-SDK is proprietary and must be purchased prior to running this program. Once the SDK libraries are built
according to the provided instructions, create a folder named `ptu-sdk` in the working directory. Then put the contents
of the PTU-SDK in this folder so that `cpi.h`, `libcpi.a`, `cerial/`, and `examples/` are directly under the `ptu-sdk/`
directory. This program uses PTU-SDK version `2.0.4`.

## opencv

1. https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html

## libcaer

1. https://gitlab.com/inivation/dv/libcaer

## To Do:
* Reimplement systematic errors

## References

1. https://gist.github.com/Yousha/3830712334ac30a90eb6041b932b68d7