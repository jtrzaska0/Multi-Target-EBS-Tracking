#pragma once

#include <future>
#include <armadillo>
#include <mlpack.hpp>
#include <utility>

double update_average(int prev_val, int new_val, int n_samples) {
    return (prev_val*n_samples + new_val)/((double)n_samples + 1);
}

bool check_move_stage(float pan_position, float prev_pan_position, float tilt_position, float prev_tilt_position, double update) {
    float pan_change = abs((pan_position - prev_pan_position)/prev_pan_position);
    float tilt_change = abs((tilt_position - prev_tilt_position)/prev_tilt_position);
    if (pan_change > update || tilt_change > update)
        return true;
    return false;
}

arma::mat positions_vector_to_matrix(std::vector<double> positions) {
    int n_positions = (int)(positions.size() / 2);
    arma::mat positions_mat;
    if (n_positions > 0) {
        positions_mat.zeros(2, n_positions);
        for(int i = 0; i < n_positions; i++) {
            positions_mat(0, i) = positions[2*i];
            positions_mat(1, i) = positions[2*i+1];
        }
    }
    return positions_mat;
}

arma::mat add_position_history(const arma::mat& position_history, arma::mat positions) {
    //Shift columns to the right and populate first column with most recent position
    arma::mat ret = arma::shift(position_history, +1, 1);
    ret(0,0) = positions(0,0);
    ret(1,0) = positions(1,0);

    return ret;
}

arma::mat get_kmeans_positions(const arma::mat& positions_mat) {
    mlpack::KMeans<> k;
    arma::Row<size_t> assignments;
    arma::mat centroids;
    int n_clusters = std::min((int)positions_mat.n_cols, 3);
    k.Cluster(positions_mat, n_clusters, assignments, centroids);

    return centroids;
}

arma::mat get_dbscan_positions(const arma::mat& positions_mat, double eps) {
    mlpack::dbscan::DBSCAN<> db(eps, 3);
    arma::Row<size_t> assignments;
    arma::mat centroids;
    db.Cluster(positions_mat, assignments, centroids);

    int n_clusters = (int)centroids.n_cols;
    if (n_clusters > 0)
        return centroids;

    // If no clusters found, send matrix of unassigned positions
    return positions_mat;
}

arma::mat run_tracker (std::vector<double> events, double dt, DBSCAN_KNN T, bool enable_tracking) {
    // double free or corruption (out)
    std::vector<double> positions;
    //double free or corruption (!prev)
    if (!enable_tracking || events.empty()) {
        return positions_vector_to_matrix(positions);
    }
    //free(): invalid pointer
    // The detector takes a pointer to events.
    double *mem{events.data()};
    // Starting time.
    double t0{events[0]};

    // Keep sizes of the vectors in variables.
    int nEvents{(int) events.size() / 4};
    while (true) {
        // Read all events in one integration time.
        double t1{t0 + dt};
        int N{0};
        for (; N < (int) (events.data() + events.size() - mem) / 4; ++N)
            if (mem[4 * N] >= t1)
                break;

        // Advance starting time.
        t0 = t1;
        // Feed events to the detector/tracker.
        if (N > 0) {
            //double free or corruption (!prev), double free or corruption (out), corrupted size vs. prev_size while consolidating
            T(mem, N);
            Eigen::MatrixXd targets{T.currentTracks()};

            // Break once all events have been used and push last positions
            if (t0 > events[4 * (nEvents - 1)]) {
                for (int i{0}; i < targets.rows(); ++i) {
                    positions.push_back(targets(i, 0));
                    positions.push_back(targets(i, 1));
                }
                break;
            }
            // Evolve tracks in time.
            T.predict();

            // Update eventIdx
            mem += 4 * N;
        }
    }
    //double free or corruption (!prev), corrupted size vs. prev_size while consolidating, free(): invalid pointer
    return positions_vector_to_matrix(positions);
}

arma::mat get_position(const std::string& method, const arma::mat& positions, arma::mat& previous_positions, double eps) {
    arma::mat ret;
    if ((int)positions.n_cols > 0) {
        if (method == "median") {
            ret = arma::median(positions, 1);
            return ret;
        }
        if (method == "median-history") {
            arma::mat latest = arma::median(positions, 1);
            previous_positions = add_position_history(previous_positions, latest);
            ret = arma::mean(previous_positions, 1);
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

std::tuple<arma::mat, std::string> calculate_window(cv::Mat cvmat, arma::mat positions, arma::mat prev_positions, double mag, int Nx, int Ny, const std::string& position_method, double eps, bool enable_tracking, bool enable_event_log) {
    int y_increment = (int)(mag * Ny / 2);
    int x_increment = (int)(y_increment * Nx / Ny);
    std::string positions_string;
    arma::mat stage_positions;
    auto start = std::chrono::high_resolution_clock::now();

    if (enable_tracking) {
        int thickness = 2;
        int x_min, x_max, y_min, y_max;
        stage_positions = get_position(position_method, positions, prev_positions, eps);

        int first_x = 0;
        int first_y = 0;
        if (stage_positions.n_cols > 0) {
            first_x = (int) positions(0, 0);
            first_y = (int) positions(1, 0);
        }

        for (int i=0; i < (int)positions.n_cols; i++) {
            int x = (int)positions(0,i);
            int y = (int)positions(1,i);
            if (x == -10 || y == -10) {
                continue;
            }

            y_min = std::max(y - y_increment, 0);
            x_min = std::max(x - x_increment, 0);
            y_max = std::min(y + y_increment, Ny - 1);
            x_max = std::min(x + x_increment, Nx - 1);

            cv::Point p1(x_min, y_min);
            cv::Point p2(x_max, y_max);
            rectangle(cvmat, p1, p2,
                      cv::Scalar(255, 0, 0),
                      thickness, cv::LINE_8);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> timestamp_ms = end - start;
        for (int i = 0; i < stage_positions.n_cols; i++) {
            int x_stage = (int)stage_positions(0, i);
            int y_stage = (int)stage_positions(1, i);
            y_min = std::max(y_stage - y_increment, 0);
            x_min = std::max(x_stage - x_increment, 0);
            y_max = std::min(y_stage + y_increment, Ny - 1);
            x_max = std::min(x_stage + x_increment, Nx - 1);

            cv::Point p1_stage(x_min, y_min);
            cv::Point p2_stage(x_max, y_max);

            rectangle(cvmat, p1_stage, p2_stage,
                      cv::Scalar(0, 0, 255),
                      thickness, cv::LINE_8);

            if(enable_event_log) {
                positions_string += std::to_string(timestamp_ms.count()) + ",";
                positions_string += std::to_string(x_stage) + ",";
                positions_string += std::to_string(y_stage) + "\n";
            }
        }

        cv::putText(cvmat,
                    std::string("Objects: ") + std::to_string((int)(stage_positions.size()/2)), //text
                    cv::Point((int)(0.05*Nx),(int)(0.95*Ny)),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5,
                    CV_RGB(118, 185, 0),
                    2);

        cv::putText(cvmat,
                    std::string("(") + std::to_string(first_x) + std::string(", ") + std::to_string(first_y) + std::string(")"), //text
                    cv::Point((int)(0.80*Nx), (int)(0.95*Ny)),
                    cv::FONT_HERSHEY_DUPLEX,
                    0.5,
                    CV_RGB(118, 185, 0),
                    2);
    }
    std::tuple<arma::mat, std::string> ret = {stage_positions, positions_string};
    return ret;
}

void update_window(const std::string& winname, const cv::Mat& cvmat) {
    if (cvmat.rows > 0 && cvmat.cols > 0 ) {
        cv::imshow(winname, cvmat);
        cv::waitKey(1);
    }
}

std::tuple<cv::Mat, std::string> read_packets(std::vector<double> events, int Nx, int Ny, bool enable_event_log) {
    std::string event_string;
    cv::Mat cvEvents(Ny, Nx, CV_8UC3, cv::Vec3b{127, 127, 127});
    if (events.empty()) {
        std::tuple<cv::Mat, std::string> ret = {cvEvents, event_string};
        return ret;
    }
    for (int i = 0; i < events.size(); i += 4) {
        double ts = events.at(i);
        int x = (int) events.at(i + 1);
        int y = (int) events.at(i + 2);
        int pol = (int) events.at(i + 3);
        cvEvents.at<cv::Vec3b>(y, x) = pol ? cv::Vec3b{255, 255, 255} : cv::Vec3b{0, 0, 0};
        if(enable_event_log) {
            event_string += std::to_string(ts) + ",";
            event_string += std::to_string(x) + ",";
            event_string += std::to_string(y) + ",";
            event_string += std::to_string(pol) + "\n";
        }
    }
    std::tuple<cv::Mat, std::string> ret = {cvEvents, event_string};
    return ret;
}

// take a packet and run through the entire processing taskflow
// return a cv mat for plotting and an arma mat for stage positions
std::tuple<cv::Mat, arma::mat, std::string, std::string> process_packet(std::vector<double> events, double dt, DBSCAN_KNN T, bool enable_tracking,
                                              int Nx, int Ny, bool enable_event_log, arma::mat prev_positions,
                                              double mag, const std::string& position_method, double eps) {
    cv::Mat event_image;
    arma::mat stage_positions;
    std::string event_string;
    std::string positions_string;
    std::future<arma::mat> fut_positions = std::async(std::launch::async, run_tracker, events, dt, T, enable_tracking);
    std::future<std::tuple<cv::Mat, std::string>> fut_event_image = std::async(std::launch::async, read_packets, events, Nx, Ny, enable_event_log);
    arma::mat positions = fut_positions.get();
    std::tie(event_image, event_string) = fut_event_image.get();
    std::tie(stage_positions, positions_string) = calculate_window(event_image, positions, std::move(prev_positions), mag, Nx, Ny, position_method, eps, enable_tracking, enable_event_log);
    std::tuple<cv::Mat, arma::mat, std::string, std::string> ret = {event_image, stage_positions, event_string, positions_string};
    return ret;
}

std::tuple<int, int, double, double, double, float, float, float, float, float, float, double> get_calibration(Stage* kessler) {
    std::tuple<int, int, double, double, double, float, float, float, float, float, float, double> cal_params;
    if (kessler) {
        std::mutex mtx;
        double r;
        kessler->handshake();
        std::cout << kessler->get_device_info().to_string();
        bool cal_active = true;
        std::thread pinger(ping, kessler, std::ref(mtx), std::ref(cal_active));
        auto const [nx, ny, hfovx, hfovy, y0, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error] = calibrate_stage(kessler);
        printf("Enter approximate target distance in meters:\n");
        std::cin >> r;
        cal_active = false;
        pinger.join();
        cal_params = {nx, ny, hfovx, hfovy, y0, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error, r};
    }
    return cal_params;
}

std::chrono::time_point<std::chrono::high_resolution_clock> move_stage (Stage* kessler, const arma::mat& positions, int nx, int ny, float begin_pan, float end_pan, float begin_tilt,
                 float end_tilt, float theta_prime_error, float phi_prime_error, double hfovx, double hfovy, double y0,
                 double r, std::chrono::time_point<std::chrono::high_resolution_clock> last_start) {
    if (kessler) {
        // Go to first position in list. Selecting between objects to be implemented later.
        double x = positions(0,0) - ((double) nx / 2);
        double y = ((double) ny / 2) - positions(1,0);

        double theta = get_theta(y, ny, hfovy);
        double phi = get_phi(x, nx, hfovx);
        double theta_prime = get_theta_prime(phi, theta, y0, r, theta_prime_error);
        double phi_prime = get_phi_prime(phi, theta, y0, r, phi_prime_error);

        float pan_position = get_pan_position(begin_pan, end_pan, phi_prime);
        float tilt_position = get_tilt_position(begin_tilt, end_tilt, theta_prime);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> timestamp_ms = end - last_start;
        if (timestamp_ms.count() > 100) {
            printf("Calculated Stage Angles: (%0.2f, %0.2f)\n", theta_prime * 180 / PI,
                   phi_prime * 180 / PI);
            printf("Stage Positions:\n     Pan: %0.2f (End: %0.2f)\n     Tilt: %0.2f (End: %0.2f)\n",
                   pan_position, end_pan - begin_pan, tilt_position, end_tilt - begin_tilt);
            printf("Moving stage to (%.2f, %.2f)\n\n", x, y);

            kessler->set_position_speed_acceleration(2, pan_position, (float)0.6*PAN_MAX_SPEED, PAN_MAX_ACC);
            kessler->set_position_speed_acceleration(3, tilt_position, (float)0.6*TILT_MAX_SPEED, TILT_MAX_ACC);

            return std::chrono::high_resolution_clock::now();
        }
    }
    return last_start;
}