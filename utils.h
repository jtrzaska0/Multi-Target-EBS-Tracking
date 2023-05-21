#pragma once

#include <future>
#include <armadillo>
#include <mlpack.hpp>
#include <utility>

using json = nlohmann::json;

class ProcessingInit {
public:
    double dt;
    bool enable_tracking;
    int Nx;
    int Ny;
    bool enable_event_log;
    std::string event_file;
    double mag;
    std::string position_method;
    double eps;
    bool report_average;
    double update;
    int update_time;
    double r_center;
    bool save_video;
    bool enable_stage;
    double hfovx;
    double hfovy;
    double offset_x;
    double offset_y;
    double offset_z;
    double arm;
    double theta_prime_error;
    double phi_prime_error;
    int begin_pan;
    int end_pan;
    int begin_tilt;
    int end_tilt;
    float begin_pan_angle;
    float end_pan_angle;
    float begin_tilt_angle;
    float end_tilt_angle;

    ProcessingInit(double dt, bool enable_tracking, int Nx, int Ny, bool enable_event_log, const std::string &event_file,
                   double mag, const std::string &position_method, double eps, bool report_average, double update,
                   int update_time, double r_center, bool save_video, bool enable_stage, double hfovx, double hfovy,
                   double offset_x, double offset_y, double offset_z, double arm, double theta_prime_error,
                   double phi_prime_error, int begin_pan, int end_pan, int begin_tilt, int end_tilt,
                   float begin_pan_angle, float end_pan_angle, float begin_tilt_angle, float end_tilt_angle) {
        this->dt = dt;
        this->enable_tracking = enable_tracking;
        this->Nx = Nx;
        this->Ny = Ny;
        this->enable_event_log = enable_event_log;
        this->event_file = event_file;
        this->mag = mag;
        this->position_method = position_method;
        this->eps = eps;
        this->report_average = report_average;
        this->update = update;
        this->update_time = update_time;
        this->r_center = r_center;
        this->save_video = save_video;
        this->enable_stage = enable_stage;
        this->hfovx = hfovx;
        this->hfovy = hfovy;
        this->offset_x = offset_x;
        this->offset_y = offset_y;
        this->offset_z = offset_z;
        this->arm = arm;
        this->theta_prime_error = theta_prime_error;
        this->phi_prime_error = phi_prime_error;
        this->begin_pan = begin_pan;
        this->end_pan = end_pan;
        this->begin_tilt = begin_tilt;
        this->end_tilt = end_tilt;
        this->begin_pan_angle = begin_pan_angle;
        this->end_pan_angle = end_pan_angle;
        this->begin_tilt_angle = begin_tilt_angle;
        this->end_tilt_angle = end_tilt_angle;
    }
};

class EventInfo {
public:
    cv::Mat event_image;
    std::string event_string;

    EventInfo() = default;

    EventInfo(cv::Mat event_image, const std::string &event_string) {
        this->event_image = std::move(event_image);
        this->event_string = event_string;
    }
};

class WindowInfo {
public:
    EventInfo event_info;
    arma::mat stage_positions;
    std::string positions_string;
    int prev_x;
    int prev_y;
    int n_samples;

    WindowInfo() {
        prev_x = 0;
        prev_y = 0;
        n_samples = 0;
    }

    WindowInfo(EventInfo event_info, arma::mat stage_positions, const std::string &positions_string, int prev_x,
               int prev_y, int n_samples) {
        this->event_info = std::move(event_info);
        this->stage_positions = std::move(stage_positions);
        this->positions_string = positions_string;
        this->prev_x = prev_x;
        this->prev_y = prev_y;
        this->n_samples = n_samples;
    }
};

class StageInfo {
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int prev_pan;
    int prev_tilt;

    StageInfo(std::chrono::time_point<std::chrono::high_resolution_clock> end, int prev_pan, int prev_tilt) {
        this->end = end;
        this->prev_pan = prev_pan;
        this->prev_tilt = prev_tilt;
    }
};

double update_average(int prev_val, int new_val, int n_samples) {
    return (prev_val * n_samples + new_val) / ((double) n_samples + 1);
}

bool check_move_stage(int pan_position, int prev_pan_position, int tilt_position, int prev_tilt_position, double update) {
    float pan_change = abs((float)(pan_position - prev_pan_position) / (float)prev_pan_position);
    float tilt_change = abs((float)(tilt_position - prev_tilt_position) / (float)prev_tilt_position);
    if (pan_change > update || tilt_change > update)
        return true;
    return false;
}

arma::mat positions_vector_to_matrix(std::vector<double> positions) {
    int n_positions = (int) (positions.size() / 2);
    arma::mat positions_mat;
    if (n_positions > 0) {
        positions_mat.zeros(2, n_positions);
        for (int i = 0; i < n_positions; i++) {
            positions_mat(0, i) = positions[2 * i];
            positions_mat(1, i) = positions[2 * i + 1];
        }
    }
    return positions_mat;
}

void add_position_history(arma::mat *position_history, arma::mat positions, std::binary_semaphore *update_positions) {
    //Shift columns to the right and populate first column with most recent position
    if (update_positions->try_acquire()) {
        arma::mat ret = arma::shift(*position_history, +1, 1);
        ret(0, 0) = positions(0, 0);
        ret(1, 0) = positions(1, 0);
        *position_history = ret;
        update_positions->release();
    }
}

arma::mat get_kmeans_positions(const arma::mat &positions_mat) {
    mlpack::KMeans<> k;
    arma::Row<size_t> assignments;
    arma::mat centroids;
    int n_clusters = std::min((int) positions_mat.n_cols, 3);
    k.Cluster(positions_mat, n_clusters, assignments, centroids);

    return centroids;
}

arma::mat get_dbscan_positions(const arma::mat &positions_mat, double eps) {
    mlpack::dbscan::DBSCAN<> db(eps, 3);
    arma::Row<size_t> assignments;
    arma::mat centroids;
    db.Cluster(positions_mat, assignments, centroids);

    int n_clusters = (int) centroids.n_cols;
    if (n_clusters > 0)
        return centroids;

    // If no clusters found, send matrix of unassigned positions
    return positions_mat;
}

arma::mat run_tracker(std::vector<double> events, double dt, DBSCAN_KNN T, bool enable_tracking) {
    std::vector<double> positions;
    if (!enable_tracking || events.empty()) {
        return positions_vector_to_matrix(positions);
    }
    double *mem{events.data()};
    double t0{events[0]};
    int nEvents{(int) events.size() / 4};
    while (true) {
        double t1{t0 + dt};
        int N{0};
        for (; N < (int) (events.data() + events.size() - mem) / 4; ++N)
            if (mem[4 * N] >= t1)
                break;

        t0 = t1;
        if (N > 0) {
            T(mem, N);
            Eigen::MatrixXd targets{T.currentTracks()};

            if (t0 > events[4 * (nEvents - 1)]) {
                for (int i{0}; i < targets.rows(); ++i) {
                    positions.push_back(targets(i, 0));
                    positions.push_back(targets(i, 1));
                }
                break;
            }
            T.predict();
            mem += 4 * N;
        }
    }
    return positions_vector_to_matrix(positions);
}

arma::mat get_position(const std::string &method, const arma::mat &positions, arma::mat *previous_positions, double eps,
                       std::binary_semaphore *update_positions) {
    arma::mat ret;
    if ((int) positions.n_cols > 0) {
        if (method == "median") {
            ret = arma::median(positions, 1);
            return ret;
        }
        if (method == "median-history") {
            arma::mat latest = arma::median(positions, 1);
            add_position_history(previous_positions, latest, update_positions);
            ret = arma::mean(*previous_positions, 1);
            return ret;
        }
        if (method == "dbscan") {
            ret = get_dbscan_positions(positions, eps);
            return ret;
        }
        if (method == "kmeans") {
            ret = get_kmeans_positions(positions);
            return ret;
        }
    }
    return ret;
}

WindowInfo calculate_window(const ProcessingInit &proc_init, const EventInfo &event_info, arma::mat positions,
                            arma::mat *prev_positions, std::binary_semaphore *update_positions, int prev_x,
                            int prev_y, int n_samples) {
    int y_increment = (int) (proc_init.mag * proc_init.Ny / 2);
    int x_increment = (int) (y_increment * proc_init.Nx / proc_init.Ny);
    std::string positions_string;
    arma::mat stage_positions;
    auto start = std::chrono::high_resolution_clock::now();

    if (proc_init.enable_tracking) {
        int thickness = 2;
        int x_min, x_max, y_min, y_max;
        stage_positions = get_position(proc_init.position_method, positions, prev_positions, proc_init.eps, update_positions);

        int first_x = 0;
        int first_y = 0;
        if (stage_positions.n_cols > 0) {
            n_samples += 1;
            first_x = (int) (stage_positions(0, 0) - ((float) proc_init.Nx / 2));
            first_y = (int) (((float) proc_init.Ny / 2) - stage_positions(1, 0));
            if (proc_init.report_average) {
                first_x = (int) update_average(prev_x, first_x, n_samples);
                first_y = (int) update_average(prev_y, first_y, n_samples);
                if (n_samples > 500)
                    n_samples = 0;
            }
        }
        prev_x = first_x;
        prev_y = first_y;

        for (int i = 0; i < (int) positions.n_cols; i++) {
            int x = (int) positions(0, i);
            int y = (int) positions(1, i);
            if (x == -10 || y == -10) {
                continue;
            }

            y_min = std::max(y - y_increment, 0);
            x_min = std::max(x - x_increment, 0);
            y_max = std::min(y + y_increment, proc_init.Ny - 1);
            x_max = std::min(x + x_increment, proc_init.Nx - 1);

            cv::Point p1(x_min, y_min);
            cv::Point p2(x_max, y_max);
            rectangle(event_info.event_image, p1, p2, cv::Scalar(255, 0, 0), thickness, cv::LINE_8);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> timestamp_ms = end - start;
        for (int i = 0; i < stage_positions.n_cols; i++) {
            int x_stage = (int) stage_positions(0, i);
            int y_stage = (int) stage_positions(1, i);
            y_min = std::max(y_stage - y_increment, 0);
            x_min = std::max(x_stage - x_increment, 0);
            y_max = std::min(y_stage + y_increment, proc_init.Ny - 1);
            x_max = std::min(x_stage + x_increment, proc_init.Nx - 1);

            cv::Point p1_stage(x_min, y_min);
            cv::Point p2_stage(x_max, y_max);
            rectangle(event_info.event_image, p1_stage, p2_stage, cv::Scalar(0, 0, 255), thickness, cv::LINE_8);

            if (proc_init.enable_event_log) {
                positions_string += std::to_string(timestamp_ms.count()) + ",";
                positions_string += std::to_string(x_stage) + ",";
                positions_string += std::to_string(y_stage) + "\n";
            }
        }

        cv::putText(event_info.event_image,
                    std::string("Objects: ") + std::to_string((int) (stage_positions.size() / 2)), //text
                    cv::Point((int) (0.05 * proc_init.Nx), (int) (0.95 * proc_init.Ny)),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5,
                    CV_RGB(118, 185, 0),
                    2);

        cv::putText(event_info.event_image,
                    std::string("(") + std::to_string(first_x) + std::string(", ") + std::to_string(first_y) +
                    std::string(")"), //text
                    cv::Point((int) (0.80 * proc_init.Nx), (int) (0.95 * proc_init.Ny)),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5,
                    CV_RGB(118, 185, 0),
                    2);
    }
    WindowInfo info(event_info, stage_positions, positions_string, prev_x, prev_y, n_samples);
    return info;
}

void update_window(const std::string &winname, const cv::Mat &cvmat) {
    if (cvmat.rows > 0 && cvmat.cols > 0) {
        cv::imshow(winname, cvmat);
        cv::waitKey(1);
    }
}

EventInfo read_packets(std::vector<double> events, int Nx, int Ny, bool enable_event_log) {
    std::string event_string;
    cv::Mat cvEvents(Ny, Nx, CV_8UC3, cv::Vec3b{127, 127, 127});
    if (events.empty()) {
        EventInfo info;
        return info;
    }
    for (int i = 0; i < events.size(); i += 4) {
        double ts = events.at(i);
        int x = (int) events.at(i + 1);
        int y = (int) events.at(i + 2);
        int pol = (int) events.at(i + 3);
        cvEvents.at<cv::Vec3b>(y, x) = pol ? cv::Vec3b{255, 255, 255} : cv::Vec3b{0, 0, 0};
        if (enable_event_log) {
            event_string += std::to_string(ts) + ",";
            event_string += std::to_string(x) + ",";
            event_string += std::to_string(y) + ",";
            event_string += std::to_string(pol) + "\n";
        }
    }
    EventInfo info(cvEvents, event_string);
    return info;
}

// take a packet and run through the entire processing taskflow
// return a cv mat for plotting and an arma mat for stage positions
WindowInfo process_packet(std::vector<double> events, DBSCAN_KNN T, const ProcessingInit &proc_init,
                          const WindowInfo &prev_window, arma::mat *prev_positions,
                          std::binary_semaphore *update_positions) {
    cv::Mat event_image;
    arma::mat stage_positions;
    std::future<arma::mat> fut_positions = std::async(std::launch::async, run_tracker, events, proc_init.dt, T,
                                                      proc_init.enable_tracking);
    std::future<EventInfo> fut_event_info = std::async(std::launch::async, read_packets, events,
                                                       proc_init.Nx, proc_init.Ny, proc_init.enable_event_log);
    arma::mat positions = fut_positions.get();
    EventInfo event_info = fut_event_info.get();
    WindowInfo tracking_info = calculate_window(proc_init, event_info, positions, prev_positions, update_positions,
                                                prev_window.prev_x, prev_window.prev_y, prev_window.n_samples);
    return tracking_info;
}

StageInfo move_stage(struct cerial *cer, const ProcessingInit &proc_init, arma::mat positions,
                     std::chrono::time_point<std::chrono::high_resolution_clock> last_start, int prev_pan,
                     int prev_tilt) {
    if (proc_init.enable_stage) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> timestamp_ms = end - last_start;
        if (positions.n_cols > 0) {
            // Go to first position in list. Selecting between objects to be implemented later.
            double x = positions(0, 0) - ((double) proc_init.Nx / 2);
            double y = ((double) proc_init.Ny / 2) - positions(1, 0);

            double theta = get_theta(y, proc_init.Ny, proc_init.hfovy);
            double phi = get_phi(x, proc_init.Nx, proc_init.hfovx);
            double theta_prime = get_theta_prime(phi, theta, proc_init.offset_x, proc_init.offset_y, proc_init.offset_z, proc_init.r_center, proc_init.arm, proc_init.theta_prime_error);
            double phi_prime = get_phi_prime(phi, proc_init.offset_x, proc_init.offset_y, proc_init.r_center, proc_init.phi_prime_error);
            int pan_position = get_motor_position(proc_init.begin_pan, proc_init.end_pan,
                                                  proc_init.begin_pan_angle, proc_init.end_pan_angle, phi_prime);
            // Convert tilt to FLIR frame
            theta_prime = M_PI_2 - theta_prime;
            int tilt_position = get_motor_position(proc_init.begin_tilt, proc_init.end_tilt,
                                                   proc_init.begin_tilt_angle, proc_init.end_tilt_angle, theta_prime);

            bool move = check_move_stage(pan_position, prev_pan, tilt_position, prev_tilt, proc_init.update);
            if (timestamp_ms.count() > proc_init.update_time && move) {
                printf("Calculated Stage Angles: (%0.2f, %0.2f)\n", theta_prime * 180 / M_PI, phi_prime * 180 / M_PI);
                printf("Stage Positions:\n     Pan: %d (%d, %d)\n     Tilt: %d (%d, %d)\n",
                       pan_position, proc_init.begin_pan, proc_init.end_pan, tilt_position,
                       proc_init.begin_tilt, proc_init.end_tilt);
                printf("Moving stage to (%.2f, %.2f)\n\n", x, y);

                uint16_t status;
                cpi_ptcmd(cer, &status, OP_PAN_DESIRED_POS_SET, pan_position);
                cpi_ptcmd(cer, &status, OP_TILT_DESIRED_POS_SET, tilt_position);
                StageInfo info(std::chrono::high_resolution_clock::now(), pan_position, tilt_position);
                return info;
            }
        }
    }
    StageInfo info(last_start, prev_pan, prev_tilt);
    return info;
}

std::tuple<StageInfo, WindowInfo>
read_future(struct cerial *cer, std::future<WindowInfo> &future, const ProcessingInit &proc_init,
            const StageInfo &prevStage,
            std::ofstream &stageFile, std::ofstream &eventFile, cv::VideoWriter &video) {
    const WindowInfo window_info = future.get();
    update_window("PLOT_EVENTS", window_info.event_info.event_image);
    stageFile << window_info.positions_string;
    eventFile << window_info.event_info.event_string;
    if (proc_init.save_video)
        video.write(window_info.event_info.event_image);
    StageInfo stage_info = move_stage(cer, proc_init, window_info.stage_positions, prevStage.end, prevStage.prev_pan,
                                      prevStage.prev_tilt);
    std::tuple<StageInfo, WindowInfo> ret = {stage_info, window_info};
    return ret;
}