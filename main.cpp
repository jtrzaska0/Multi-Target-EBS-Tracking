#include <fstream>
#include <nlohmann/json.hpp>
#include "Event-Sensor-Detection-and-Tracking/Algorithm.hpp"
#include "threads.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {
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
        HISTORY_SIZE: Number of previous positions to average in history
        MAX_SPEED: Number between 0 and 1. Sets percent of max speed to move the stage
        SAVE_VIDEO: true or false
        VIDEO_FILEPATH: File for video. Include the ".mp4" extension
        VIDEO_FPS: Target video frames per second

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
    const int history_size = params.value("HISTORY_SIZE", 12);
    double max_speed = params.value("MAX_SPEED", 0.6);
    double max_acc = params.value("MAX_ACCELERATION", 1);
    bool enable_filter = noise_params.value("ENABLE_FILTER", false);
    bool save_video = params.value("SAVE_VIDEO", false);
    std::string video_file = params.value("VIDEO_FILEPATH", "output.mp4");
    int video_fps = params.value("VIDEO_FPS", 30);
    double focal_len = stage_params.value("FOCAL_LENGTH", 0.006);
    double sep = stage_params.value("SEPARATION", 0.15);
    double dist = stage_params.value("DISTANCE", 10);
    double px_size = stage_params.value("PIXEL_SIZE", 0.000009);
    bool correction = stage_params.value("SYSTEMATIC_ERROR", false);
    bool prev_cal = stage_params.value("USE_PREVIOUS", false);
    float begin_pan_angle = (float) stage_params.value("START_PAN_ANGLE", -M_PI_2);
    float end_pan_angle = (float) stage_params.value("END_PAN_ANGLE", M_PI_2);
    float begin_tilt_angle = (float) stage_params.value("START_TILT_ANGLE", 0);
    float end_tilt_angle = (float) stage_params.value("END_TILT_ANGLE", 2 * M_PI / 3);
    Buffers buffers(history_size);

    /**Create an Algorithm object here.**/
    // Matrix initializer
    // DBSCAN
    Eigen::MatrixXd invals{Eigen::MatrixXd::Zero(1, 4)};

    // Mean Shift
    invals(0, 0) = 5.2;
    invals(0, 1) = 9;
    invals(0, 2) = 74;
    invals(0, 3) = 1.2;
    // Model initializer
    double DT = integrationtime;
    double p3 = pow(DT, 3) / 3;
    double p2 = pow(DT, 2) / 2;

    Eigen::MatrixXd P{{16, 0,  0, 0},
                      {0,  16, 0, 0},
                      {0,  0,  9, 0},
                      {0,  0,  0, 9}};
    Eigen::MatrixXd F{{1, 0, DT, 0},
                      {0, 1, 0,  DT},
                      {0, 0, 1,  0},
                      {0, 0, 0,  1}};
    Eigen::MatrixXd Q{{p3, 0,  p2, 0},
                      {0,  p3, 0,  p2},
                      {p2, 0,  DT, 0},
                      {0,  p2, 0,  DT}};
    Eigen::MatrixXd H{{1, 0, 0, 0},
                      {0, 1, 0, 0}};
    Eigen::MatrixXd R{{7, 0},
                      {0, 7}};

    // Define the model.
    KModel k_model{.dt = DT, .P = P, .F = F, .Q = Q, .H = H, .R = R};
    // Algo initializer
    DBSCAN_KNN algo(invals, k_model);

    int Nx = 640;
    int Ny = 480;
    if (device_type == "davis") {
        Nx = 346;
        Ny = 260;
    }

    int ret;
    Calibration cal_params;
    float cal_dist;
    bool active = true;
    CalibrationInit cal_init(focal_len, sep, dist, px_size, Nx, Ny, correction, prev_cal, begin_pan_angle,
                             end_pan_angle, begin_tilt_angle, end_tilt_angle, XK_C);
    ProcessingInit proc_init(max_speed, max_acc, DT, enable_tracking, Nx, Ny,
                             enable_event_log, event_file, mag, position_method,
                             eps, report_average, stage_update, update_time, cal_dist, save_video);
    cv::startWindowThread();
    cv::namedWindow("PLOT_EVENTS",
                    cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO |
                    cv::WindowFlags::WINDOW_GUI_EXPANDED);
    cv::VideoWriter video(video_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), video_fps, cv::Size(Nx, Ny));

    if (enable_stage) {
        Stage stage("192.168.50.1", 5520);
        stage.handshake();
        std::cout << stage.get_device_info().to_string();
        std::tie(cal_params, cal_dist) = get_calibration(&stage, cal_init);
        std::thread processor(processing_threads, std::ref(buffers), &stage, algo, std::ref(video), std::ref(proc_init),
                              std::ref(active));
        if (device_type == "xplorer")
            ret = read_xplorer(buffers, noise_params, enable_filter, active);
        else
            ret = read_davis(buffers, noise_params, enable_filter, active);
        processor.join();
    } else {
        std::tie(cal_params, cal_dist) = get_calibration(nullptr, cal_init);
        std::thread processor(processing_threads, std::ref(buffers), nullptr, algo, std::ref(video),
                              std::ref(proc_init), std::ref(active));
        if (device_type == "xplorer")
            ret = read_xplorer(buffers, noise_params, enable_filter, active);
        else
            ret = read_davis(buffers, noise_params, enable_filter, active);
        processor.join();
    }
    cv::destroyWindow("PLOT_EVENTS");
    return ret;
}