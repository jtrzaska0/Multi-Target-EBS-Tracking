#include <fstream>
#include <nlohmann/json.hpp>
#include "Event-Sensor-Detection-and-Tracking/Algorithm.hpp"
#include "threads.h"

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    /*

    Args:
        argv[1]: Absolute path to config JSON file

    config.json:
        DEVICE_TYPE: "xplorer" or "davis"
        INTEGRATION_TIME_MS: Integration time in milliseconds.
        PACKET_NUMBER: Number of event packets to aggregate.
        ENABLE_TRACKING: true or false
        ENABLE_STAGE: true or false
        COMMAND_METHOD: Stage command calculation method: "median", "dbscan", or "median-history"
        STAGE_UPDATE: Percent change required for stage to update positions
        EPSILON: epsilon for mlpack clustering
        MAGNIFICATION: Magnification
        ENABLE_LOGGING: true or false
        EVENT_FILEPATH: File for event CSV. Do not include the ".csv" extension
        VERBOSE: Print queue sizes, true or false.
        BUFFER_SIZE: Number of elements in circular buffer
        HISTORY_SIZE: Number of previous positions to average in history
        MAX_SPEED: Number between 0 and 1. Sets percent of max speed to move the stage

    Ret:
        0
    */

    if (argc != 2) {
        printf("Invalid number of arguments.\n");
        return 1;
    }

    std::string config_file = {std::string(argv[1])};
    std::ifstream f(config_file);
    json settings = json::parse(f);
    json params = settings["PROGRAM_PARAMETERS"];
    json noise_params = settings["NOISE_FILTER"];
    json stage_params = settings["STAGE_SETTINGS"];

    std::string device_type = params.value("DEVICE_TYPE", "xplorer");
    double integrationtime = params.value("INTEGRATION_TIME_MS", 2);
    bool enable_tracking = params.value("ENABLE_TRACKING", false);
    bool enable_stage = stage_params.value("ENABLE_STAGE", false);
    std::string position_method = params.value("COMMAND_METHOD", "median-history");
    double eps = params.value("EPSILON", 15);
    double mag = params.value("MAGNIFICATION", 0.05);
    bool enable_event_log = params.value("ENABLE_LOGGING", false);
    std::string event_file = params.value("EVENT_FILEPATH", "recording");
    double stage_update = stage_params.value("STAGE_UPDATE", 0.02);
    int update_time = stage_params.value("UPDATE_TIME", 100);
    bool report_average = params.value("REPORT_AVERAGE", false);
    bool verbose = params.value("VERBOSE", false);
    const int buffer_size = params.value("BUFFER_SIZE", 100);
    const int history_size = params.value("HISTORY_SIZE", 12);
    double max_speed = params.value("MAX_SPEED", 0.6);
    double max_acc = params.value("MAX_ACCELERATION", 1);
    bool enable_filter = noise_params.value("ENABLE_FILTER", false);
    Buffers buffers(buffer_size, history_size);

    /**Create an Algorithm object here.**/
    // Matrix initializer
    // DBSCAN
    Eigen::MatrixXd invals {Eigen::MatrixXd::Zero(1, 4)};

    // Mean Shift
    invals(0,0) = 5.2;
    invals(0,1) = 9;
    invals(0,2) = 74;
    invals(0,3) = 1.2;
    // Model initializer
    double DT = integrationtime;
    double p3 = pow(DT, 3) / 3;
    double p2 = pow(DT, 2) / 2;

    Eigen::MatrixXd P {{16, 0, 0, 0}, {0, 16, 0, 0}, {0, 0, 9, 0}, {0, 0, 0, 9}};
    Eigen::MatrixXd F {{1, 0, DT, 0}, {0, 1, 0, DT}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Eigen::MatrixXd Q {{p3, 0, p2, 0}, {0, p3, 0, p2}, {p2, 0, DT, 0}, {0, p2, 0, DT}};
    Eigen::MatrixXd H {{1, 0, 0, 0}, {0, 1, 0, 0}};
    Eigen::MatrixXd R {{7, 0}, {0, 7}};

    // Define the model.
    KModel k_model {.dt = DT, .P = P, .F = F, .Q = Q, .H = H, .R = R};
    // Algo initializer
    DBSCAN_KNN algo(invals, k_model);

    int Nx = 640;
    int Ny = 480;
    if (device_type == "davis") {
        Nx = 346;
        Ny = 260;
    }

    int ret;
    bool active = true;
    cv::startWindowThread();
    cv::namedWindow("PLOT_EVENTS",
                    cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);

    if (enable_stage) {
        Stage stage("192.168.50.1", 5520);
        stage.handshake();
        std::cout << stage.get_device_info().to_string();
        std::tuple<int, int, double, double, double, float, float, float, float, float, float, double, float, float, float, float> cal_params = get_calibration(&stage, stage_params);
        std::thread processor(processing_threads, std::ref(buffers), &stage, max_speed, max_acc, DT, algo, enable_tracking, Nx, Ny, enable_event_log, event_file, mag, position_method, eps, report_average, stage_update, update_time, std::ref(active), cal_params);
        if (device_type == "xplorer")
            ret = read_xplorer(buffers, noise_params, verbose, enable_filter, active);
        else
            ret = read_davis(buffers, noise_params, verbose, enable_filter, active);
        processor.join();
    }
    else {
        std::tuple<int, int, double, double, double, float, float, float, float, float, float, double, float, float, float, float> cal_params = get_calibration(nullptr, stage_params);
        std::thread processor(processing_threads, std::ref(buffers), nullptr, max_speed, max_acc, DT, algo, enable_tracking, Nx, Ny, enable_event_log, event_file, mag, position_method, eps, report_average, stage_update, update_time, std::ref(active), cal_params);
        if (device_type == "xplorer")
            ret = read_xplorer(buffers, noise_params, verbose, enable_filter, active);
        else
            ret = read_davis(buffers, noise_params, verbose, enable_filter, active);
        processor.join();
    }

    cv::destroyWindow("PLOT_EVENTS");
    return ret;
}