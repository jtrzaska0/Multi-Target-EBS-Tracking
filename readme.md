# Installation
## mlpack
First, install the dependencies:
1. `sudo apt-get install liblapack-dev`
2. `sudo apt-get install libblas-dev`
3.  `sudo apt-get install libboost-dev`
4. `sudo apt-get install libarmadillo-dev`
5. `sudo apt-get install libensmallen-dev`
6. `sudo apt-get install libcereal-dev`

Then, `mlpack` should be built and installed from according to the Github page: https://github.com/mlpack/mlpack
1. `git clone https://github.com/mlpack/mlpack`
2. `cd mlpack`
3. `mkdir build && cd build/`
4. `cmake ..`
5. `sudo make install`

## conan
1. `conan install . -pr:b=default --output-folder=build --build=missing`

## Event Sensor Detection and Tracking
1. Clone the Event Sensor Detection and Tracking repo into the working directory
   * `cd ...\LiveTracking`
   * `gh repo clone I2SL/Event-Sensor-Detection-and-Tracking`

## opencv
1. https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html

## libcaer
1. https://gitlab.com/inivation/dv/libcaer

## References
1. https://gist.github.com/Yousha/3830712334ac30a90eb6041b932b68d7