#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/stat.h>

extern "C" {
#include "ptu-sdk/examples/estrap.h"
}

#include "stage-camera/StageCam.h"
#include "Event-Sensor-Detection-and-Tracking/Algorithm.hpp"
#include "threads.h"
#include "controller.h"
#include "videos.h"

using json = nlohmann::json;

bool directoryExists(const std::string& folderPath) {
    struct stat info{};
    return stat(folderPath.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool deleteDirectory(const std::string& directoryPath) {
    if (!std::filesystem::exists(directoryPath)) {
        std::cerr << "Directory does not exist: " << directoryPath << std::endl;
        return false;
    }

    try {
        std::filesystem::remove_all(directoryPath);
        std::cout << "Directory deleted: " << directoryPath << std::endl;
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Failed to delete directory: " << e.what() << std::endl;
        return false;
    }
}

bool makeDirectory(const std::string& directoryPath) {
    if (directoryExists(directoryPath)) {
        if (!deleteDirectory(directoryPath))
            return false;
    }
    if (mkdir(directoryPath.c_str(), 0777) == -1) {
        std::cerr << "Failed to create directory." << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    /*

    Args:
        argv[1]: -p
        argv[2]: tcp:<FLIR stage IP address>
        argv[3]: Path to config JSON file
        argv[4]: Path to ONNX file

    config.json:
        See readme for variable descriptions.

    Ret:
        0
    */

    std::string config_file = {std::string(argv[3])};
    std::string onnx_loc = {std::string(argv[4])};
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
    double r_center = stage_params.value("OBJECT_DIST", 999999.9);
    bool enable_event_log = params.value("ENABLE_LOGGING", false);
    std::string event_file = params.value("EVENT_FILEPATH", "recording");
    int stage_update = stage_params.value("STAGE_UPDATE", 5);
    int update_time = stage_params.value("UPDATE_TIME", 100);
    bool report_average = params.value("REPORT_AVERAGE", false);
    const int history_size = params.value("HISTORY_SIZE", 12);
    bool enable_filter = noise_params.value("ENABLE_FILTER", false);
    int video_fps = params.value("VIDEO_FPS", 30);
    double focal_len = stage_params.value("FOCAL_LENGTH", 0.006);
    double offset_x = stage_params.value("OFFSET_X", 0.0);
    double offset_y = stage_params.value("OFFSET_Y", 0.15);
    double offset_z = stage_params.value("OFFSET_Z", 0.0);
    double arm = stage_params.value("ARM_LENGTH", 0.2);
    double dist = stage_params.value("FOCUS_DIST", 999999.9);
    float begin_pan_angle = (float) stage_params.value("START_PAN_ANGLE", -M_PI_2);
    float end_pan_angle = (float) stage_params.value("END_PAN_ANGLE", M_PI_2);
    float begin_tilt_angle = (float) stage_params.value("START_TILT_ANGLE", -M_PI / 6);
    float end_tilt_angle = (float) stage_params.value("END_TILT_ANGLE", M_PI / 6);
    int max_tilt_pos = stage_params.value("MAX_TILT_POS", 1500);
    int min_tilt_pos = stage_params.value("MIN_TILT_POS", -1500);
    int max_pan_pos = stage_params.value("MAX_PAN_POS", 4500);
    int min_pan_pos = stage_params.value("MIN_PAN_POS", -4500);
    int max_tilt_speed = stage_params.value("MAX_TILT_SPEED", 6000);
    int min_tilt_speed = stage_params.value("MIN_TILT_SPEED", 0);
    int max_pan_speed = stage_params.value("MAX_PAN_SPEED", 6000);
    int min_pan_speed = stage_params.value("MIN_PAN_SPEED", 0);
    int pan_acc = stage_params.value("PAN_ACC", 6000);
    int tilt_acc = stage_params.value("TILT_ACC", 6000);
    double nfov_focal_len = stage_params.value("NFOV_FOCAL_LENGTH", 0.100);
    int nfov_nx = stage_params.value("NFOV_NPX_HORIZ", 2592);
    int nfov_ny = stage_params.value("NFOV_NPX_VERT", 1944);
    double nfov_px_size = stage_params.value("NFOV_PX_PITCH", 0.0000022);
    double confidence_thres = params.value("CONFIDENCE", 0.4);
    double kp_fine = params.value("KP_FINE", 0.0);
    double ki_fine = params.value("KI_FINE", 0.0);
    double kd_fine = params.value("KD_FINE", 0.0);
    double kp_coarse = params.value("KP_COARSE", 0.0);
    double ki_coarse = params.value("KI_COARSE", 0.0);
    double kd_coarse = params.value("KD_COARSE", 0.0);
    bool enable_dnn = params.value("ENABLE_DNN", true);
    bool enable_pid = params.value("ENABLE_PID", true);
    int tilt_offset = stage_params.value("TILT_OFFSET", 0);
    int pan_offset = stage_params.value("PAN_OFFSET", 0);
    double ebs_eps = params.value("EBS_EPSILON", 8.0);
    int ebs_num_pts = params.value("EBS_NUM_PTS", 8);
    double ebs_tau = params.value("EBS_TAU", 1.2);
    bool verbose = params.value("VERBOSE", false);
    double coarse_overshoot_time = stage_params.value("COARSE_OVERSHOOT_TIME", 0.2);
    double fine_overshoot_time = stage_params.value("FINE_OVERSHOOT_TIME", 0.2);
    int overshoot_thres = stage_params.value("OVERSHOOT_THRESHOLD", 100);
    bool debug = params.value("DEBUG", false);
    Buffers buffers(history_size);

    // DBSCAN
    Eigen::MatrixXd invals {Eigen::MatrixXd::Zero(1, 3)};
    invals(0, 0) = ebs_eps;
    invals(0, 1) = ebs_num_pts;
    invals(0, 2) = ebs_tau;

    // Model initializer
    double DT = integrationtime;
    double p3 = pow(DT, 3) / 3;
    double p2 = pow(DT, 2) / 2;

    Eigen::MatrixXd P {{16, 0, 0, 0}, {0, 16, 0, 0}, {0, 0, 9, 0}, {0, 0, 0, 9}};
    Eigen::MatrixXd F {{1, 0, DT, 0}, {0, 1, 0, DT}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Eigen::MatrixXd Q {{p3, 0, p2, 0}, {0, p3, 0, p2}, {p2, 0, DT, 0}, {0, p2, 0, DT}};
    Eigen::MatrixXd H {{1, 0, 0, 0}, {0, 1, 0, 0}};
    Eigen::MatrixXd R {{7, 0}, {0, 7}};

    // Define the model for KNN.
    KModel k_model {.dt = DT, .P = P, .F = F, .Q = Q, .H = H, .R = R};
    // Algo initializer
    DBSCAN_KNN algo(invals, k_model);

    int Nx = 640;
    int Ny = 480;
    double px_size = 0.000009;
    if (device_type == "davis") {
        Nx = 346;
        Ny = 260;
        px_size = 0.0000185;
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
        if (cpi_ptcmd(cer, &status, OP_PAN_USER_MAX_POS_SET, max_pan_pos) ||
            cpi_ptcmd(cer, &status, OP_PAN_USER_MIN_POS_SET, min_pan_pos) ||
            cpi_ptcmd(cer, &status, OP_TILT_USER_MAX_POS_SET, max_tilt_pos) ||
            cpi_ptcmd(cer, &status, OP_TILT_USER_MIN_POS_SET, min_tilt_pos) ||
            cpi_ptcmd(cer, &status, OP_TILT_LOWER_SPEED_LIMIT_SET, min_tilt_speed) ||
            cpi_ptcmd(cer, &status, OP_TILT_UPPER_SPEED_LIMIT_SET, max_tilt_speed) ||
            cpi_ptcmd(cer, &status, OP_PAN_LOWER_SPEED_LIMIT_SET, min_pan_speed) ||
            cpi_ptcmd(cer, &status, OP_PAN_UPPER_SPEED_LIMIT_SET, max_pan_speed) ||
            cpi_ptcmd(cer, &status, OP_PAN_ACCEL_SET, pan_acc) ||
            cpi_ptcmd(cer, &status, OP_TILT_ACCEL_SET, tilt_acc))
            die("Basic unit queries failed.\n");
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    StageController ctrl(kp_coarse, ki_coarse, kd_coarse, kp_fine, ki_fine, kd_fine, 4500, -4500, 1500, -1500, start_time,
                         event_file, enable_event_log, cer, enable_pid, fine_overshoot_time, coarse_overshoot_time,
                         overshoot_thres, update_time, stage_update, verbose);

    int cam_width = 640;
    int cam_height = 480;
    double nfov_hfovx = get_hfov(nfov_focal_len, dist, nfov_nx, nfov_px_size);
    double nfov_hfovy = get_hfov(nfov_focal_len, dist, nfov_ny, nfov_px_size);
    StageCam stageCam(cam_width, cam_height);
    cv::namedWindow("Camera", cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO |
                              cv::WindowFlags::WINDOW_GUI_EXPANDED);

    ProcessingInit proc_init(DT, enable_tracking, Nx, Ny, enable_event_log, event_file, mag, position_method, eps,
                             report_average, r_center, enable_stage, hfovx, hfovy, offset_x, offset_y, offset_z, arm,
                             pan_offset, tilt_offset, min_pan_pos, max_pan_pos, min_tilt_pos, max_tilt_pos,
                             begin_pan_angle, end_pan_angle, begin_tilt_angle, end_tilt_angle, verbose);
    cv::startWindowThread();
    cv::namedWindow("PLOT_EVENTS", cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO |
                                   cv::WindowFlags::WINDOW_GUI_EXPANDED);

    if (!makeDirectory("./event_images") || !makeDirectory("./camera_images"))
        return -1;

    std::thread processor(processing_threads, std::ref(ctrl), std::ref(buffers), algo, std::ref(proc_init), start_time, debug, std::ref(active));
    std::thread camera(camera_thread, std::ref(stageCam), std::ref(ctrl), cam_height, cam_width, nfov_hfovx, nfov_hfovy, onnx_loc, enable_stage, enable_dnn, start_time, confidence_thres, debug, std::ref(active));
    if (device_type == "xplorer")
        ret = read_xplorer(buffers, debug, noise_params, enable_filter, event_file, start_time, active);
    else
        ret = read_davis(buffers, debug, noise_params, enable_filter, event_file, start_time, active);

    processor.join();
    camera.join();
    ctrl.shutdown();
    if (cer) {
        cpi_ptcmd(cer, &status, OP_PAN_DESIRED_SPEED_SET, 9000);
        cpi_ptcmd(cer, &status, OP_TILT_DESIRED_SPEED_SET, 9000);
        cpi_ptcmd(cer, &status, OP_PAN_DESIRED_POS_SET, 0);
        cpi_ptcmd(cer, &status, OP_TILT_DESIRED_POS_SET, 0);
    }
    cv::destroyAllWindows();
    printf("Processing camera video...\n");
    createVideoFromImages("./camera_images", "camera_output.mp4", video_fps);
    printf("Processing event video...\n");
    createVideoFromImages("./event_images", "ebs_output.mp4", video_fps);
    return ret;
}