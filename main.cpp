#include <fstream>
#include <nlohmann/json.hpp>
extern "C" {
#include "ptu-sdk/examples/estrap.h"
}
#include "Event-Sensor-Detection-and-Tracking/Algorithm.hpp"
#include "threads.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {
    /*

    Args:
        argv[1]: Absolute path to config JSON file
        argv[2]: -p
        argv[3]: tcp:<FLIR stage IP address>

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

    std::string config_file = {std::string(argv[3])};
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
    float cal_dist = (float)params.value("OBJECT_DIST", 999999.9);
    bool enable_event_log = params.value("ENABLE_LOGGING", false);
    std::string event_file = params.value("EVENT_FILEPATH", "recording");
    double stage_update = stage_params.value("STAGE_UPDATE", 0.02);
    int update_time = stage_params.value("UPDATE_TIME", 100);
    bool report_average = params.value("REPORT_AVERAGE", false);
    const int history_size = params.value("HISTORY_SIZE", 12);
    bool enable_filter = noise_params.value("ENABLE_FILTER", false);
    bool save_video = params.value("SAVE_VIDEO", false);
    std::string video_file = params.value("VIDEO_FILEPATH", "output.mp4");
    int video_fps = params.value("VIDEO_FPS", 30);
    double focal_len = stage_params.value("FOCAL_LENGTH", 0.006);
    double sep = stage_params.value("SEPARATION", 0.15);
    double dist = stage_params.value("DISTANCE", 10);
    double px_size = stage_params.value("PIXEL_SIZE", 0.000009);
    bool correction = stage_params.value("SYSTEMATIC_ERROR", false);
    float begin_pan_angle = (float) stage_params.value("START_PAN_ANGLE", -M_PI_2);
    float end_pan_angle = (float) stage_params.value("END_PAN_ANGLE", M_PI_2);
    float begin_tilt_angle = (float) stage_params.value("START_TILT_ANGLE", -M_PI / 6);
    float end_tilt_angle = (float) stage_params.value("END_TILT_ANGLE", M_PI / 6);
    int max_tilt_pos = params.value("MAX_TILT_POS", 1500);
    int min_tilt_pos = params.value("MIN_TILT_POS", -1500);
    int max_pan_pos = params.value("MAX_PAN_POS", 4500);
    int min_pan_pos = params.value("MIN_PAN_POS", -4500);
    int max_tilt_speed = params.value("MAX_TILT_SPEED", 6000);
    int min_tilt_speed = params.value("MIN_TILT_SPEED", 0);
    int max_pan_speed = params.value("MAX_PAN_SPEED", 6000);
    int min_pan_speed = params.value("MIN_PAN_SPEED", 0);
    int pan_acc = params.value("PAN_ACC", 6000);
    int tilt_acc = params.value("TILT_ACC", 6000);
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
    bool active = true;
    double hfovx = get_hfov(focal_len, dist, Nx, px_size);
    double hfovy = get_hfov(focal_len, dist, Ny, px_size);

    struct cerial *cer = nullptr;
    uint16_t status;
    if (enable_stage) {
        if ((cer = estrap_in(argc, argv)) == nullptr) {
            printf("Failed to connect to stage.\n");
            return 1;
        }
        // Set terse mode
        if (cpi_ptcmd(cer, &status, OP_FEEDBACK_SET, CPI_ASCII_FEEDBACK_TERSE))
            die("Failed to set feedback mode.\n");

        // Set min/max positions, speed, and acceleration
        if (cpi_ptcmd(cer, &status, OP_PAN_USER_MAX_POS_SET, max_pan_pos) || cpi_ptcmd(cer, &status, OP_PAN_USER_MIN_POS_SET, min_pan_pos) ||
            cpi_ptcmd(cer, &status, OP_TILT_USER_MAX_POS_SET, max_tilt_pos) || cpi_ptcmd(cer, &status, OP_TILT_USER_MIN_POS_SET, min_tilt_pos) ||
            cpi_ptcmd(cer, &status, OP_TILT_LOWER_SPEED_LIMIT_SET, min_tilt_speed) || cpi_ptcmd(cer, &status, OP_TILT_UPPER_SPEED_LIMIT_SET, max_tilt_speed) ||
            cpi_ptcmd(cer, &status, OP_PAN_LOWER_SPEED_LIMIT_SET, min_pan_speed) || cpi_ptcmd(cer, &status, OP_PAN_UPPER_SPEED_LIMIT_SET, max_pan_speed) ||
            cpi_ptcmd(cer, &status, OP_PAN_ACCEL_SET, pan_acc) || cpi_ptcmd(cer, &status, OP_TILT_ACCEL_SET, tilt_acc))
            die("Basic unit queries failed.\n");

        printf("Min Pan: %0.2f deg\nMax Pan: %0.2f deg\n", min_pan_pos * 0.02, max_pan_pos * 0.02);
        printf("Min Tilt: %0.2f deg\nMax Tilt: %0.2f deg\n", min_tilt_pos * 0.02, max_tilt_pos * 0.02);
    }

    double theta_prime_error{}, phi_prime_error{};
    if (correction && enable_stage) {
        std::thread driver(controller, cer, XK_C);
        int pan_position = 0;
        int tilt_position = 0;
        double r;
        double x;
        double y;
        printf("Center camera on known target and press `Q`.\n");
        while (true) {
            bool q_pressed = key_is_pressed(XK_Q);
            if (q_pressed) {
                cpi_ptcmd(cer, &status, OP_PAN_CURRENT_POS_GET, &pan_position);
                cpi_ptcmd(cer, &status, OP_TILT_CURRENT_POS_GET, &tilt_position);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        printf("Enter target distance (m):\n");
        std::cin >> r;
        printf("Enter target x coordinate:\n");
        std::cin >> x;
        printf("Enter target y coordinate:\n");
        std::cin >> y;

        double phi = get_phi(x, Nx, hfovx);
        double theta = get_theta(y, Ny, hfovy);
        double phi_prime_estimate = get_phi_prime(phi, theta, sep, r, 0);
        double theta_prime_estimate = get_theta_prime(phi, theta, sep, r, 0);
        theta_prime_estimate = M_PI_2 - theta_prime_estimate;
        double phi_prime_actual = (double)pan_position * M_PI / 9000.0;
        double theta_prime_actual = (double)tilt_position * M_PI / 9000.0;
        phi_prime_error = (phi_prime_actual - phi_prime_estimate);
        theta_prime_error = (theta_prime_actual - theta_prime_estimate);
        printf("Estimated theta_p/phi_p: (%.2f, %.2f)\n", theta_prime_estimate * 180 / M_PI, phi_prime_estimate * 180 / M_PI);
        printf("Actual theta_p/phi_p: (%.2f, %.2f)\n", theta_prime_actual * 180 / M_PI, phi_prime_actual * 180 / M_PI);
        printf("Calibration complete. Press C to exit manual control.\n");
        driver.join();
    }

    ProcessingInit proc_init(DT, enable_tracking, Nx, Ny, enable_event_log, event_file, mag, position_method, eps,
                             report_average, stage_update, update_time, cal_dist, save_video, enable_stage, hfovx, hfovy,
                             sep, theta_prime_error, phi_prime_error, min_pan_pos, max_pan_pos, min_tilt_pos, max_tilt_pos, begin_pan_angle,
                             end_pan_angle, begin_tilt_angle, end_tilt_angle);
    cv::startWindowThread();
    cv::namedWindow("PLOT_EVENTS", cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);
    cv::VideoWriter video(video_file, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), video_fps, cv::Size(Nx, Ny));
    std::thread processor(processing_threads, cer, std::ref(buffers), algo, std::ref(video), std::ref(proc_init), std::ref(active));
    if (device_type == "xplorer")
        ret = read_xplorer(buffers, noise_params, enable_filter, active);
    else
        ret = read_davis(buffers, noise_params, enable_filter, active);
    processor.join();

    cv::destroyWindow("PLOT_EVENTS");
    return ret;
}