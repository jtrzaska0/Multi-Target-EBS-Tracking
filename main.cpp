// File     main.cpp
// Summary  Read in program configuration and start up the tracking loops.
// Authors  Trevor Schlackt, Jacob Trzaska

# include <filesystem>
# include <vector>
# include <string>
# include <fstream>
# include <cassert>
# include <atomic>
# include <nlohmann/json.hpp>
# include <sys/stat.h>

extern "C" {
# include "ptu-sdk/examples/estrap.h"
}

# include "stage-camera/StageCam.h"
# include "Event-Sensor-Detection-and-Tracking/Algorithm.hpp"
# include "threads.h"
# include "controller.h"
# include "videos.h"

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
        std::cerr << "Directory deleted: " << directoryPath << std::endl;
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

class stream_redirection {
    /*
    Redirect a stream. This implementation was taken from an answer by 'Toby Speight' to
    the question ""Why segment fault occurs when redirect outstream to a file." on 
    StackOverflow.
    */

    private:
    std::ostream& from;
    std::ofstream to;
    std::streambuf * const saved;

    public:
    stream_redirection(std::ostream& from, const std::string& filename) : 
        from{from}, to {filename}, saved{from.rdbuf(to.rdbuf())} {
        /*
        Perform the redirection.
        */

        return;
    }

    stream_redirection(const stream_redirection&) = delete;

    void operator=(const stream_redirection&) = delete;

    ~stream_redirection() {
        /*
        Cleanup.
        */

        from.rdbuf(saved);
    }
};



int main(int argc, char ** argv) {
    /*

    Args:
        argv[1]: Path to config JSON file
        argv[2]: Path to ONNX file

    config.json:
        See readme for variable descriptions.

    Ret:
        0
    */

    // We'll use ncurses to control program behavior. This requires
    // keeping controlling writes to stdout, which is partially
    // accomplished by redirecting standard error.
    std::ofstream err("err.log", std::ios::trunc | std::ios::out);
    if (!err.good()) {
        std::cout << "Could not create log file.\n";
        return -1;

    }

    auto redirect {stream_redirection(std::cerr, "err.log")};

    // Locate onnx file.
    std::string onnx_loc = {std::string(argv[2])};

    // Locate, load, and parse the json configuration file.
    std::string config_file = {std::string(argv[1])};
    std::ifstream f(config_file);
    json settings = json::parse(f);
    json params = settings["PROGRAM_PARAMETERS"];
    json noise_params = settings["NOISE_FILTER"];
    json stage_params = settings["STAGE_SETTINGS"];

    // Set program (fine-track, EBS, etc) configuration.
    std::string device_type = params.value("DEVICE_TYPE", "xplorer");              
    double integrationtime = params.value("INTEGRATION_TIME_MS", 2);               
    bool enable_tracking = params.value("ENABLE_TRACKING", false);                 
    std::string position_method = params.value("COMMAND_METHOD", "median-history");
    double eps = params.value("EPSILON", 15);                                      
    double mag = params.value("MAGNIFICATION", 0.05);
    bool enable_event_log = params.value("ENABLE_LOGGING", false);                 
    std::string event_file = params.value("EVENT_FILEPATH", "recording");          
    bool report_average = params.value("REPORT_AVERAGE", false);                   
    const int history_size = params.value("HISTORY_SIZE", 12);                     
    bool enable_filter = noise_params.value("ENABLE_FILTER", false);               
    double ebs_eps = params.value("EBS_EPSILON", 8.0);                             
    int ebs_num_pts = params.value("EBS_NUM_PTS", 8);                              
    double ebs_tau = params.value("EBS_TAU", 1.2);                                 
    bool verbose = params.value("VERBOSE", false);
    bool debug = params.value("DEBUG", false);
    Buffers buffers(history_size);
    int num_stages = params.value("NUM_STAGES",  0);

    if (num_stages <= 0 || num_stages >= 100) {
        std::cerr << "Invalid number of stages. Edit 'config.json'. Aborting.\n";
        exit(EXIT_SUCCESS);
    }

    // Set the stage configuration(s).
    std::vector<bool> enable_stage(num_stages);
    std::vector<double> r_center(num_stages);
    std::vector<int> stage_update(num_stages);
    std::vector<int> update_time(num_stages);
    std::vector<double> focal_len(num_stages);
    std::vector<double> offset_x(num_stages);
    std::vector<double> offset_y(num_stages);
    std::vector<double> offset_z(num_stages);
    std::vector<double> arm(num_stages);
    std::vector<double> dist(num_stages);
    std::vector<float> begin_pan_angle(num_stages);
    std::vector<float> end_pan_angle(num_stages);
    std::vector<float> begin_tilt_angle(num_stages);
    std::vector<float> end_tilt_angle(num_stages);
    std::vector<int> max_tilt_pos(num_stages);
    std::vector<int> min_tilt_pos(num_stages);
    std::vector<int> max_pan_pos(num_stages);
    std::vector<int> min_pan_pos(num_stages);
    std::vector<int> max_tilt_speed(num_stages);
    std::vector<int> min_tilt_speed(num_stages);
    std::vector<int> max_pan_speed(num_stages);
    std::vector<int> min_pan_speed(num_stages);
    std::vector<int> pan_acc(num_stages);
    std::vector<int> tilt_acc(num_stages);
    std::vector<double> nfov_focal_len(num_stages);
    std::vector<int> nfov_nx(num_stages);
    std::vector<int> nfov_ny(num_stages);
    std::vector<double> nfov_px_size(num_stages);
    std::vector<int> tilt_offset(num_stages);
    std::vector<int> pan_offset(num_stages);
    std::vector<double> coarse_overshoot_time(num_stages);
    std::vector<double> fine_overshoot_time(num_stages);
    std::vector<int> overshoot_thres(num_stages);
    std::vector<std::string> tcp_ip_addr(num_stages);

    // Set the program's vector configurations.
    std::vector<double> kp_fine(num_stages);     
    std::vector<double> ki_fine(num_stages);     
    std::vector<double> kd_fine(num_stages);     
    std::vector<double> kp_coarse(num_stages); 
    std::vector<double> ki_coarse(num_stages); 
    std::vector<double> kd_coarse(num_stages); 
    std::vector<std::atomic<bool>> enable_dnn(num_stages);
    std::vector<bool> enable_pid(num_stages);
    std::vector<int> video_fps(num_stages);
    std::vector<double> confidence_thres(num_stages);

    for (int n {0}; n < num_stages; ++n) {
        // Use ASCII encoding to index the configuration vectors. Note that
        // this also dictates our manner of naming the keys in the config
        // file.
        char Index[3] {'0', '0', '\0'};
        char * index {Index};

        if (n < 10) {
            Index[1] = 48 + n;
            index = Index + 1;
        } else {
            Index[0] = n / 10;
            Index[1] = n % 10;
        }

        // Store the n-th stage configuration.
        enable_stage[n] = stage_params["ENABLE_STAGE"].value(index, false);
        r_center[n] = stage_params["OBJECT_DIST"].value(index, 999999.9);
        stage_update[n] = stage_params["STAGE_UPDATE"].value(index, 5);
        update_time[n] = stage_params["UPDATE_TIME"].value(index, 100);
        focal_len[n] = stage_params["FOCAL_LENGTH"].value(index, 0.006);
        offset_x[n] = stage_params["OFFSET_X"].value(index, 0.0);
        offset_y[n] = stage_params["OFFSET_Y"].value(index, 0.15);
        offset_z[n] = stage_params["OFFSET_Z"].value(index, 0.0);
        arm[n] = stage_params["ARM"].value(index, 0.2);
        dist[n] = stage_params["FOCUS_DIST"].value(index, 999999.9);
        begin_pan_angle[n] = (float) stage_params["START_PAN_ANGLE"].value(index, -M_PI_2);
        end_pan_angle[n] = (float) stage_params["END_PAN_ANGLE"].value(index, M_PI_2);
        begin_tilt_angle[n] = (float) stage_params["START_TILT_ANGLE"].value(index, -M_PI / 6);
        end_tilt_angle[n] = (float) stage_params["END_TILT_ANGLE"].value(index, M_PI / 6);
        max_tilt_pos[n] = stage_params["MAX_TILT_POS"].value(index, 1500);
        min_tilt_pos[n] = stage_params["MIN_TILT_POS"].value(index, -1500);
        max_pan_pos[n] = stage_params["MAX_PAN_POS"].value(index, 4500);
        min_pan_pos[n] = stage_params["MIN_PAN_POS"].value(index, -4500);
        max_tilt_speed[n] = stage_params["MAX_TILT_SPEED"].value(index, 6000);
        min_tilt_speed[n] = stage_params["MIN_TILT_SPEED"].value(index, 0);
        max_pan_speed[n] = stage_params["MAX_PAN_SPEED"].value(index, 6000);
        min_pan_speed[n] = stage_params["MIN_PAN_SPEED"].value(index, 0);
        pan_acc[n] = stage_params["PAN_ACC"].value(index, 6000);
        tilt_acc[n] = stage_params["TILT_ACC"].value(index, 6000);
        nfov_focal_len[n] = stage_params["NFOV_FOCAL_LENGTH"].value(index, 0.100);
        nfov_nx[n] = stage_params["NFOV_NPX_HORIZ"].value(index, 2592);
        nfov_ny[n] = stage_params["NFOV_NPX_VERT"].value(index, 1944);
        nfov_px_size[n] = stage_params["NFOV_PX_PITCH"].value(index, 0.0000022);
        tilt_offset[n] = stage_params["TILT_OFFSET"].value(index, 0);
        pan_offset[n] = stage_params["PAN_OFFSET"].value(index, 0);
        coarse_overshoot_time[n] = stage_params["COARSE_OVERSHOOT_TIME"].value(index, 0.2);
        fine_overshoot_time[n] = stage_params["FINE_OVERSHOOT_TIME"].value(index, 0.2);
        overshoot_thres[n] = stage_params["OVERSHOOT_THRESHOLD"].value(index, 100);
        tcp_ip_addr[n] = stage_params["TCP_IP_INIT"].value(index, "null");

        if (tcp_ip_addr[n] == "null") {
            std::cerr << "Invalid IP address. Abort.\n";
            exit(EXIT_SUCCESS);
        }

        // Store the stage configurations.
        kp_fine[n] = params["KP_FINE"].value(index, 0.0);     
        ki_fine[n] = params["KI_FINE"].value(index, 0.0);     
        kd_fine[n] = params["KD_FINE"].value(index, 0.0);     
        kp_coarse[n] = params["KP_COARSE"].value(index, 0.0); 
        ki_coarse[n] = params["KI_COARSE"].value(index, 0.0); 
        kd_coarse[n] = params["KD_COARSE"].value(index, 0.0); 
        enable_dnn[n] = params["ENABLE_DNN"].value(index, true);
        enable_pid[n] = params["ENABLE_PID"].value(index, true);
        video_fps[n] = params["VIDEO_FPS"].value(index, 30);
        confidence_thres[n] = params["CONFIDENCE"].value(index, 0.4);
    }

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

    int Nx {(device_type == "davis") ? 346 : 640};
    int Ny {(device_type == "davis") ? 260 : 480};
    double px_size {(device_type == "davis") ? 18.5e-6 : 9e-6};

    int ret;
    bool active = true;
    double hfovx {get_hfov(focal_len[0], dist[0], Nx, px_size)}; // We should only require one focal length.
    double hfovy {get_hfov(focal_len[0], dist[0], Ny, px_size)};

    // Connect to the stage(s).
    std::vector<struct cerial *> cer(num_stages);
    std::vector<uint16_t> status(num_stages);

    for (int n {0}; n < num_stages; ++n) {
        if (enable_stage[n]) {
            char p[3] {'-', 'p', '\0'};
            char * tcp_ip {(char *)tcp_ip_addr[n].c_str()};
            char * estrap_in_input[3] {argv[0], p, tcp_ip};

            // Feed argc_n, argv_n to estrap.
            if ((cer[n] = estrap_in(3, estrap_in_input)) == nullptr) {
                std::cerr << "Failed to connect to stage " + std::to_string(n) + ".\n";
                return 1;
            }

            // Set terse mode
            if (cpi_ptcmd(cer[n], &status[n], OP_FEEDBACK_SET, CPI_ASCII_FEEDBACK_TERSE))
                die("Failed to set feedback mode.\n");
    
            // Set min/max positions, speed, and acceleration
            if (cpi_ptcmd(cer[n], &status[n], OP_PAN_USER_MAX_POS_SET, max_pan_pos[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_PAN_USER_MIN_POS_SET, min_pan_pos[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_TILT_USER_MAX_POS_SET, max_tilt_pos[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_TILT_USER_MIN_POS_SET, min_tilt_pos[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_TILT_LOWER_SPEED_LIMIT_SET, min_tilt_speed[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_TILT_UPPER_SPEED_LIMIT_SET, max_tilt_speed[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_PAN_LOWER_SPEED_LIMIT_SET, min_pan_speed[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_PAN_UPPER_SPEED_LIMIT_SET, max_pan_speed[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_PAN_ACCEL_SET, pan_acc[n]) ||
                cpi_ptcmd(cer[n], &status[n], OP_TILT_ACCEL_SET, tilt_acc[n]))
                die("Basic unit queries failed.\n");
        }
    }

    // Begin startup.
    auto start_time = std::chrono::high_resolution_clock::now();

    // Set controllers for each stage.
    std::vector<StageController *> stageControllers;
    for (int n {0}; n < num_stages; ++n) {
        StageController * ptr {new StageController(kp_coarse[n], ki_coarse[n], kd_coarse[n], kp_fine[n], ki_fine[n], kd_fine[n], 4500, -4500, 1500, -1500, start_time,
                             event_file, enable_event_log, cer[n], enable_pid[n], fine_overshoot_time[n], coarse_overshoot_time[n],
                             overshoot_thres[n], update_time[n], stage_update[n], verbose, n)
        };
 
        stageControllers.push_back(ptr);
    }

    // Ready the cameras on each stage.
    std::vector<StageCam *> stageCams;
    std::vector<double> nfov_hfovx(num_stages);
    std::vector<double> nfov_hfovy(num_stages);
    std::vector<int> cam_widths(num_stages);
    std::vector<int> cam_heights(num_stages);

    for (int n {0}; n < num_stages; ++n) {
        // Calculate field of views.
        nfov_hfovx[n] = get_hfov(nfov_focal_len[n], dist[n], nfov_nx[n], nfov_px_size[n]);
        nfov_hfovy[n] = get_hfov(nfov_focal_len[n], dist[n], nfov_ny[n], nfov_px_size[n]);

        // Calculate the string-based index.
        char Index[3] {'0', '0', '\0'};
        char * index {Index};

        if (n < 10) {
            Index[1] = 48 + n;
            index = Index + 1;
        } else {
            Index[0] = n / 10;
            Index[1] = n % 10;
        }

        // Startup camera. 
        int cam_width {stage_params["CAM_WIDTH"].value(index, 0)};
        int cam_height {stage_params["CAM_HEIGHT"].value(index, 0)};
        assert(cam_width != 0);
        assert(cam_height != 0);
        cam_widths[n] = cam_width;
        cam_heights[n] = cam_height;

        StageCam * ptr {new StageCam(cam_width, cam_height, 30000, n)}; // Likely needs to be edited to use device ID instead of index.
        stageCams.push_back(ptr);

        cv::namedWindow("Camera" + std::to_string(n), cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO |
                                  cv::WindowFlags::WINDOW_GUI_EXPANDED);

        // Could not create directories for data storage.
        if (!makeDirectory("./camera" + std::to_string(n) +"_images")) {
            std::cerr << "Failed to create frame storage for a camera. Aborting.\n";
            exit(EXIT_SUCCESS);
        }
    }
    
    // Show events in a new window.
    cv::startWindowThread();
    cv::namedWindow("PLOT_EVENTS", 
        cv::WindowFlags::WINDOW_AUTOSIZE | 
        cv::WindowFlags::WINDOW_KEEPRATIO |
        cv::WindowFlags::WINDOW_GUI_EXPANDED
    );

    if (!makeDirectory("./event_images")) {
        std::cerr << "Failed to make the event directory. Aborting.\n";
        exit(EXIT_SUCCESS);
    }
 
    // Start up the tracking loop.
    ProcessingInit proc_init(DT, enable_tracking, Nx, Ny, enable_event_log, event_file, mag, position_method, eps,
        report_average, r_center, enable_stage, hfovx, hfovy, offset_x, offset_y, offset_z, arm,
        pan_offset, tilt_offset, min_pan_pos, max_pan_pos, min_tilt_pos, max_tilt_pos,
        begin_pan_angle, end_pan_angle, begin_tilt_angle, end_tilt_angle, verbose
    );

    std::thread processor(processing_threads, std::ref(stageControllers), std::ref(buffers), 
        algo, std::ref(proc_init), std::ref(enable_dnn), start_time, debug, std::ref(active));

    // Start up the cameras.
    std::vector<std::thread *> cameraThreads;
    for (int n {0}; n < num_stages; ++n) {
        std::thread * ptr {new std::thread(
            camera_thread, 
            stageCams[n], 
            stageControllers[n], 
            cam_heights[n], cam_widths[n], 
            nfov_hfovx[n], nfov_hfovy[n], 
            std::ref(onnx_loc), enable_stage[n], 
            std::ref(enable_dnn[n]), start_time, confidence_thres[n], debug, std::ref(active), n)
        };
        
        cameraThreads.push_back(ptr);
    }

    // Start collecting events.
    if (device_type == "xplorer")
        ret = read_xplorer(buffers, debug, noise_params, enable_filter, event_file, start_time, active);
    else
        ret = read_davis(buffers, debug, noise_params, enable_filter, event_file, start_time, active);

    assert(ret == EXIT_SUCCESS);

    // Wait for all threads to terminate.
    processor.join();
    for (auto& th : cameraThreads)
        th->join();

    // Begin shutdown.
    for (int n {0}; n < num_stages; ++n) {
        // Shut down the controller.
        delete stageControllers[n];
     
        // Reset the stages.
        if (cer[n]) {
            cpi_ptcmd(cer[n], &status[n], OP_PAN_DESIRED_SPEED_SET, 9000);
            cpi_ptcmd(cer[n], &status[n], OP_TILT_DESIRED_SPEED_SET, 9000);
            cpi_ptcmd(cer[n], &status[n], OP_PAN_DESIRED_POS_SET, 0);
            cpi_ptcmd(cer[n], &status[n], OP_TILT_DESIRED_POS_SET, 0);
        }

        // Shutdown the cameras.
        delete stageCams[n];
    }    
 
    // Give the stages time to shutdown.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Stop video streaming and save movies.
    cv::destroyAllWindows();
    bool ebs_video {false};
    bool fbs_video {false};

    if (ebs_video) {
        std::cerr << "Processing event video..\n";
        for (int n {0}; n < num_stages; ++n)
            createVideoFromImages("./event_images", "ebs_output.mp4", video_fps[n]);
    }

    if (fbs_video) {
        std::cerr << "Processing camera videos.\n";
        for (int n {0}; n < num_stages; ++n)
            createVideoFromImages("./camera" + std::to_string(n) + "_images", "camera" + std::to_string(n) + "_output.mp4", video_fps[n]);
    }

    return ret;
}
