# Introduction

This document describes the structure of the tracking software. The code's objects, helper functions, main functions,
and workflow.

# Files

## `controller.h`

Contains the `PIDController` and `StageController` class definitions.

## `pointing.h`

Contains helper functions related to stage movement/position calculation:

* `double get_hfov()`
* `bool key_is_pressed()`
* `double get_phi()`
* `double get_theta()`
* `double get_phi_prime()`
* `double get_theta_prime()`
* `int get_motor_position()`

## `progressbar.h`

External code used to display a progress bar.

* https://github.com/gipert/progressbar

## `threads.h`

Contains the `Buffers` class definition, as well as helper functions related to EBS I/O and data preprocessing:

* `static void globalShutdownSignalHandler()`
* `static void usbShutdownHandler()`
* `cv::Mat formatYolov5()`

Also contains functions for collecting data from the EBS, processing data from the EBS, and processing data from the
frame-based camera. These functions are called on separate threads.

* `int read_xplorer()`

  Read events from the DVXplorer and push them to a `Buffers` object for processing.
* `int read_davis()`

  Read events from the DAVIS346 and push them to a `Buffers` object for processing.
* `void processing_threads()`

  Process event data from the `Buffers` object with three asynchronous threads.
* `void camera_thread()`

  Process most recent frame obtained by the frame-based camera.

## `utils.h`

Contains several class definitions and helper functions used throughout the project.

**Class Definitions:**

* `ProcessingInit`
* `EventInfo`
* `WindowInfo`
* `StageInfo`

**Functions:**

* `Eigen::MatrixXd armaToEigen()`
* `double update_average()`
* `arma::mat positions_vector_to_matrix()`
* `void add_position_history()`
* `arma::mat get_kmeans_positions()`
* `arma::mat get_dbscan_positions()`
* `arma::mat run_tracker()`
* `arma::mat get_position()`
* `WindowInfo calculate_window()`
* `void update_window()`
* `EventInfo read_packets()`
* `WindowInfo process_packet()`
* `StageInfo move_stage()`
* `std::tuple<StageInfo, WindowInfo> read_future()`

## `videos.h`

Contains helper functions related to saving images and creating videos:
* `void saveImage()`
* `int extractFileNameAsInt()`
* `std::pair<int, int> getMinMaxTimes()`
* `void createVideoFromImages()`

## `main.cpp`

In addition to `int main()`, this file contains helper functions related to directory manipulation:
* `bool directoryExists()`
* `bool deleteDirectory()`
* `bool makeDirectory()`

# Objects

## `StageCam`

This object handles connecting to and reading data from an Allied Vision USB camera. The constructor 
(`StageCam(int width, int height)`) sets the size of output images in pixels. When the `StageCam` object is constructed,
the `aquire()` function is called to run on a separate thread. This function connects to the first available camera and
begins continuous image acquisition. The `acquire()` function will run until `disconnect()` is called. When a new frame 
is obtained, it is stored in the `last_frame` variable as a `cv::Mat`. It can be accessed using the `get_frame()` 
method.

## `ProcessingInit`

This object consolidates important information into a single datatype in order to reduce the number of arguments
required in several functions throughout the program. The following variables can be accessed through the 
`ProcessingInit` object:
* `float dt`: integration time in ms
* `bool enable_tracking`: true if EBS tracking algorithm is enabled
* `int Nx`: horizontal pixels on EBS detector
* `int Ny`: vertical pixels on EBS detector
* `bool enable_event_log`: true if event logging is enabled
* `std::string& event_file`: filepath to event log
* `double mag`: size of object in EBS/size of object in FBS (only used for making graphics)
* `std::string& position_method`: method to calculate target position from tracking algorithm results
* `double eps`: epsilon for DBSCAN outside the EBS tracking algorithm
* `bool report_average`: true if time-averaged position is shown on output window
* `double r_center`: distance to object plane in meters
* `bool enable_stage`: true if stage is enabled
* `double hfovx`: horizontal EBS HFoV in radians
* `double hfovy`: vertical EBS HFoV in radians
* `double offset_x`: x offset from stage center of rotation to center of EBS detector in meters (see thesis for geometry)
* `double offset_y`: y offset from stage center of rotation to center of EBS detector in meters (see thesis for geometry)
* `double offset_z`: z offset from stage center of rotation to center of EBS detector in meters (see thesis for geometry)
* `double arm`: distance from stage center of rotation to center of FBS detector in meters (see thesis for geometry)
* `int pan_offset`: pan systematic error in steps
* `int tilt_offset`: tilt systematic error in steps
* `int begin_pan`: minimum pan position in steps
* `int end_pan`: maximum pan position in steps
* `int begin_tilt`: minimum tilt position in steps
* `int end_tilt`: maximum tilt position in steps
* `int begin_pan_angle`: minimum pan position in radians (see thesis for geometry)
* `int end_pan_angle`: maximum pan position in radians (see thesis for geometry)
* `int begin_tilt_angle`: minimum tilt position in radians (see thesis for geometry)
* `int end_tilt_angle`: maximum tilt position in radians (see thesis for geometry)
* `bool verbose`: true if verbosity is enabled

## `Buffers`

The `Buffers` object holds the `PacketQueue` and `prev_positions` variables. `PacketQueue` is a lock-free SPSC queue
from the Boost library. Event packets read from the EBS are pushed into this queue in the form of `std::vector<double>` 
to await for processing with the tracking algorithm.

`prev_positions` is an `arma::mat` of size (2, `history_size`). It stores previous object positions, which may be used
when determining the next object position.

## `PIDController`

This object implements a simple PID controller. It is constructed with the three proportionality constants, as well as
lower and upper bounds for the result. The `calculate()` method performs the PID calculation given the setpoint, current
position, and time interval since the last call. Error does not accumulate if the result is outside the given 
boundaries. The gains can be updated with the `update_gains()` method, and the accumulated error can be reset with
`reset()`.

## `StageController`

The `StageController` object is responsible for handling stage movement and logging. It works similarly to `StageCam`
in that it launches a control loop on a separate thread, which will run until the `shutdown()` method is called. The
`StageController` object handles pan and tilt commands simultaneously. 

The constructor takes the following arguments:
* `double kp_coarse`: proportional gain for coarse tracking
* `double ki_coarse`: integral gain for coarse tracking
* `double kd_coarse`: derivative gain for coarse tracking
* `double kp_fine`: proportional gain for fine tracking
* `double ki_fine`: integral gain for fine tracking
* `double kd_fine`: derivative gain for fine tracking
* `int pan_max`: maximum pan position in steps
* `int pan_min`: minimum pan position in steps
* `tilt_max`: maximum tilt position in steps
* `tilt_min`: minimum tilt position in steps
* `std::chrono::time_point<std::chrono::high_resolution_clock> start`: time when stage initiated
* `std::string& event_file`: filepath for logging
* `bool enable_logging`: whether to enable logging
* `cerial* cer`: connection to stage
* `bool pid`: whether to enable PID controllers
* `double fine_time`: overshoot time in milliseconds during fine track (see Note 1)
* `double coarse_time`: overshoot time in milliseconds during coarse track (see Note 1)
* `double overshoot_thres`: maximum threshold in steps for an overshoot to be added to the motion command (see Note 1)
* `int update_time`: minimum time in milliseconds that must pass between commands to move the stage
* `int update_thres`: minimum deviation from current position (in steps) required to move the stage
* `bool verbose`: whether to print stage motion information

Motion control is performed in the `ctrl_loop()` function. While the connection is active, the user can adjust the
pan and tilt offset by pressing the arrow keys. If `pid` is true, motion commands are calculated using `PIDController`
objects for the pan and tilt directions. Otherwise, the stage moves directly to a given setpoint. A motion command is
only issued if a certain time has passed since the last command (`update_time`) and the motion is large enough 
(`update_thres`). If logging is enabled, the motion command is written to a CSV with the elapsed time and whether fine
or coarse tracking was active.

The stage is moved by calling the `update_setpoints()` or `increment_setpoints()` methods. `update_setpoints()`
overwrites the current pan and tilt setpoints with the provided values. This is used during coarse tracking. If fine
tracking is active, calling `update_setpoints()` does nothing. Likewise, `increment_setpoints()` adds the provided
values to the current pan and tilt setpoints. This is used during fine tracking. If fine tracking is not active, calling
`increment_setpoints()` does nothing.

The `activate_fine()` and `deactive_fine()` methods must be called accordingly when fine tracking is 
activated/deactivated in the main program.

**Note 1:** Pan/tilt overshoot is a value (in steps) added to the pan/tilt setpoints to account for object motion during
tracking. It is calculated by multiplying the current stage velocity in steps/ms by the user-provided values of
`fine_time` and `coarse_time`. Overshoot is only added if the original change in position is sufficiently small (less 
than `overshoot_thres`). Ideal values for `fime_time` and `coarse_time` will depend on object distance, speed, and
trajectory. Overshoot corrections can be disabled by setting `fine_time` and `coarse_time` to 0 or by setting 
`overshoot_thres` to 0.

## `EventInfo`

The `EventInfo` object stores an `event_image` and `event_string` variable. `event_image` is a `cv::Mat` which
represents an event packet as a grayscale image. `event_string` is a string (ts, x, y, pol) of all events in the packet, 
which is later written to a CSV if logging is enabled.

## `WindowInfo`

The `WindowInfo` object is used for the graphical output of the program and logging tracking results. It contains an
`EventInfo` object, potential object locations as an `arma::mat` (for stage movement) and as a string (for logging),
previous object location (x, y), and the number of samples to use if average reporting is active.

## `StageInfo`

The `StageInfo` class holds the previous pan and tilt position of the stage in steps.

# Helper Functions

# Main Functions

# Workflow