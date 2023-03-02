#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <queue>
#include <semaphore>
#include <boost/circular_buffer.hpp>
#include <nlohmann/json.hpp>
#include <libcaercpp/devices/dvxplorer.hpp>
#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/filters/dvs_noise.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <X11/keysym.h>
#include <kessler/stage.h>
#include <kessler/tools/calibrator.h>

#include "utils.h"

using json = nlohmann::json;

static std::atomic_bool globalShutdown(false);
const int buffer_size = 20;

boost::circular_buffer<std::vector<double>> PlottingPacketQueue(buffer_size);
boost::circular_buffer<std::vector<double>> TrackingVectorQueue(buffer_size);
boost::circular_buffer<cv::Mat> CVMatrixQueue(buffer_size);
boost::circular_buffer<std::vector<double>> PositionsVectorQueue(buffer_size);
boost::circular_buffer<arma::mat> PlotPositionsMatrixQueue(buffer_size);
boost::circular_buffer<arma::mat> StagePositionsMatrixQueue(buffer_size);
arma::mat previous_positions_plot(2, 12, arma::fill::zeros);
arma::mat previous_positions_stage(2, 12, arma::fill::zeros);
std::counting_semaphore<1> prepareStage(0);

static void globalShutdownSignalHandler(int signal) {
    // Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for global shutdown.
    if (signal == SIGTERM || signal == SIGINT) {
        globalShutdown.store(true);
    }
}

static void usbShutdownHandler(void *ptr) {
    (void) (ptr); // UNUSED.

    globalShutdown.store(true);
}

int read_xplorer (int num_packets, bool enable_tracking, const json& noise_params, const bool& active) {
    // Install signal handler for global shutdown.
#if defined(_WIN32)
    if (signal(SIGTERM, &globalShutdownSignalHandler) == SIG_ERR) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGTERM. Error: %d.", errno);
		return (EXIT_FAILURE);
	}

	if (signal(SIGINT, &globalShutdownSignalHandler) == SIG_ERR) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGINT. Error: %d.", errno);
		return (EXIT_FAILURE);
	}
#else
    struct sigaction shutdownAction{};

    shutdownAction.sa_handler = &globalShutdownSignalHandler;
    shutdownAction.sa_flags   = 0;
    sigemptyset(&shutdownAction.sa_mask);
    sigaddset(&shutdownAction.sa_mask, SIGTERM);
    sigaddset(&shutdownAction.sa_mask, SIGINT);

    if (sigaction(SIGTERM, &shutdownAction, nullptr) == -1) {
        libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                          "Failed to set signal handler for SIGTERM. Error: %d.", errno);
        return (EXIT_FAILURE);
    }

    if (sigaction(SIGINT, &shutdownAction, nullptr) == -1) {
        libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                          "Failed to set signal handler for SIGINT. Error: %d.", errno);
        return (EXIT_FAILURE);
    }
#endif

    // Open a DAVIS, give it a device ID of 1, and don't care about USB bus or SN restrictions.
    auto handle = libcaer::devices::dvXplorer(1);

    // Let's take a look at the information we have on the device.
    auto xplorer_info = handle.infoGet();

    printf("%s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d, Logic: %d.\n", xplorer_info.deviceString,
           xplorer_info.deviceID, xplorer_info.deviceIsMaster, xplorer_info.dvsSizeX, xplorer_info.dvsSizeY,
           xplorer_info.logicVersion);

    // Send the default configuration before using the device.
    // No configuration is sent automatically!
    handle.sendDefaultConfig();

    // Add full-sized software filter to reduce DVS noise.
    libcaer::filters::DVSNoise dvsNoiseFilter = libcaer::filters::DVSNoise(xplorer_info.dvsSizeX, xplorer_info.dvsSizeY);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TWO_LEVELS, noise_params.value("CAER_BACKGROUND_ACTIVITY_TWO_LEVELS", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_CHECK_POLARITY, noise_params.value("CAER_BACKGROUND_ACTIVITY_CHECK_POLARITY", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MIN, noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MIN", 2));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MAX, noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MAX", 8));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME, noise_params.value("CAER_BACKGROUND_ACTIVITY_TIME", 2000));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE, noise_params.value("CAER_BACKGROUND_ACTIVITY_ENABLE", true));

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME, noise_params.value("CAER_REFRACTORY_PERIOD_TIME", 200));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE, noise_params.value("CAER_REFRACTORY_PERIOD_ENABLE", true));

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_ENABLE, noise_params.value("CAER_HOTPIXEL_ENABLE", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_LEARN, noise_params.value("CAER_HOTPIXEL_LEARN", true));

    // Now let's get start getting some data from the device. We just loop in blocking mode,
    // no notification needed regarding new events. The shutdown notification, for example if
    // the device is disconnected, should be listened to.
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);

    // Let's turn on blocking data-get mode to avoid wasting resources.
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

    int processed = 0;
    std::vector<double> events;
    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
        if (packetContainer == nullptr) {
            continue;
        }
        for (auto &packet: *packetContainer) {
            if (packet == nullptr) {
                continue; // Skip if nothing there.
            }

            if (packet->getEventType() == POLARITY_EVENT) {
                std::shared_ptr<libcaer::events::PolarityEventPacket> polarity
                        = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                dvsNoiseFilter.apply(*polarity);

                for (const auto &e: *polarity) {
                    // Discard invalid events (filtered out).
                    if (!e.isValid()) {
                        continue;
                    }

                    events.push_back((double) e.getTimestamp() / 1000);
                    events.push_back(e.getX());
                    events.push_back(e.getY());
                    events.push_back(e.getPolarity());
                }
                processed += 1;
                if (processed >= num_packets) {
                    if (enable_tracking) {
                        TrackingVectorQueue.push_back(events);
                    }
                    PlottingPacketQueue.push_back(events);
                    events.clear();
                    processed = 0;
                }
            }
        }
    }
    handle.dataStop();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

int read_davis (int num_packets, bool enable_tracking, const json& noise_params, const bool& active) {
    // Install signal handler for global shutdown.
#if defined(_WIN32)
    if (signal(SIGTERM, &globalShutdownSignalHandler) == SIG_ERR) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGTERM. Error: %d.", errno);
		return (EXIT_FAILURE);
	}

	if (signal(SIGINT, &globalShutdownSignalHandler) == SIG_ERR) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGINT. Error: %d.", errno);
		return (EXIT_FAILURE);
	}
#else
    struct sigaction shutdownAction{};

    shutdownAction.sa_handler = &globalShutdownSignalHandler;
    shutdownAction.sa_flags   = 0;
    sigemptyset(&shutdownAction.sa_mask);
    sigaddset(&shutdownAction.sa_mask, SIGTERM);
    sigaddset(&shutdownAction.sa_mask, SIGINT);

    if (sigaction(SIGTERM, &shutdownAction, nullptr) == -1) {
        libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                          "Failed to set signal handler for SIGTERM. Error: %d.", errno);
        return (EXIT_FAILURE);
    }

    if (sigaction(SIGINT, &shutdownAction, nullptr) == -1) {
        libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
                          "Failed to set signal handler for SIGINT. Error: %d.", errno);
        return (EXIT_FAILURE);
    }
#endif

    // Open a DAVIS, give it a device ID of 1, and don't care about USB bus or SN restrictions.
    libcaer::devices::davis davisHandle = libcaer::devices::davis(1);

    // Let's take a look at the information we have on the device.
    struct caer_davis_info davis_info = davisHandle.infoGet();

    printf("%s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d, Logic: %d.\n", davis_info.deviceString,
           davis_info.deviceID, davis_info.deviceIsMaster, davis_info.dvsSizeX, davis_info.dvsSizeY,
           davis_info.logicVersion);

    // Send the default configuration before using the device.
    // No configuration is sent automatically!
    davisHandle.sendDefaultConfig();

    // Enable hardware filters if present.
    if (davis_info.dvsHasBackgroundActivityFilter) {
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_BACKGROUND_ACTIVITY_TIME, noise_params.value("DAVIS_BACKGROUND_ACTIVITY_TIME", 8));
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_BACKGROUND_ACTIVITY, noise_params.value("DAVIS_BACKGROUND_ACTIVITY", true));

        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_REFRACTORY_PERIOD_TIME, noise_params.value("DAVIS_REFRACTORY_PERIOD_TIME", 1));
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_REFRACTORY_PERIOD, noise_params.value("DAVIS_REFRACTORY_PERIOD", true));
    }

    // Add full-sized software filter to reduce DVS noise.
    libcaer::filters::DVSNoise dvsNoiseFilter = libcaer::filters::DVSNoise(davis_info.dvsSizeX, davis_info.dvsSizeY);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TWO_LEVELS, noise_params.value("CAER_BACKGROUND_ACTIVITY_TWO_LEVELS", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_CHECK_POLARITY, noise_params.value("CAER_BACKGROUND_ACTIVITY_CHECK_POLARITY", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MIN, noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MIN", 2));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MAX, noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MAX", 8));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME, noise_params.value("CAER_BACKGROUND_ACTIVITY_TIME", 2000));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE, noise_params.value("CAER_BACKGROUND_ACTIVITY_ENABLE", true));

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME, noise_params.value("CAER_REFRACTORY_PERIOD_TIME", 200));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE, noise_params.value("CAER_REFRACTORY_PERIOD_ENABLE", true));

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_ENABLE, noise_params.value("CAER_HOTPIXEL_ENABLE", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_LEARN, noise_params.value("CAER_HOTPIXEL_LEARN", true));

    // Now let's get start getting some data from the device. We just loop in blocking mode,
    // no notification needed regarding new events. The shutdown notification, for example if
    // the device is disconnected, should be listened to.
    davisHandle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);

    // Let's turn on blocking data-get mode to avoid wasting resources.
    davisHandle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

    // Disable APS (frames) and IMU, not used for showing event filtering.
    davisHandle.configSet(DAVIS_CONFIG_APS, DAVIS_CONFIG_APS_RUN, false);
    davisHandle.configSet(DAVIS_CONFIG_IMU, DAVIS_CONFIG_IMU_RUN_ACCELEROMETER, false);
    davisHandle.configSet(DAVIS_CONFIG_IMU, DAVIS_CONFIG_IMU_RUN_GYROSCOPE, false);
    davisHandle.configSet(DAVIS_CONFIG_IMU, DAVIS_CONFIG_IMU_RUN_TEMPERATURE, false);

    int processed = 0;
    std::vector<double> events;
    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = davisHandle.dataGet();
        if (packetContainer == nullptr) {
            continue;
        }

        for (auto &packet: *packetContainer) {
            if (packet == nullptr) {
                continue; // Skip if nothing there.
            }

            if (packet->getEventType() == POLARITY_EVENT) {
                std::shared_ptr<libcaer::events::PolarityEventPacket> polarity
                        = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                dvsNoiseFilter.apply(*polarity);

                for (const auto &e: *polarity) {
                    // Discard invalid events (filtered out).
                    if (!e.isValid()) {
                        continue;
                    }

                    events.push_back((double) e.getTimestamp() / 1000);
                    events.push_back(e.getX());
                    events.push_back(e.getY());
                    events.push_back(e.getPolarity());
                }
                processed += 1;
                if (processed >= num_packets) {
                    if (enable_tracking) {
                        TrackingVectorQueue.push_back(events);
                    }
                    PlottingPacketQueue.push_back(events);
                    events.clear();
                    processed = 0;
                }
            }
        }
    }
    davisHandle.dataStop();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

void tracker (double dt, DBSCAN_KNN T, bool enable_tracking, const bool&active) {
    if (enable_tracking) {
        stop:
        while (active) {
            while (!TrackingVectorQueue.empty()) {
                if (!active)
                    goto stop;
                auto events = TrackingVectorQueue.front();
                std::vector<double> positions;
                if (!events.empty()) {
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
                }
                PositionsVectorQueue.push_back(positions);
                TrackingVectorQueue.pop_front();
            }
        }
    }
}

void positions_vector_to_matrix(const bool& active) {
    stop:
    while (active) {
        while (!PositionsVectorQueue.empty()) {
            if (!active)
                goto stop;
            auto positions = PositionsVectorQueue.front();
            int n_positions = (int)(positions.size() / 2);
            arma::mat positions_mat;
            if (n_positions > 0) {
                positions_mat.zeros(2, n_positions);
                for(int i = 0; i < n_positions; i++) {
                    positions_mat(0, i) = positions[2*i];
                    positions_mat(1, i) = positions[2*i+1];
                }
            }
            StagePositionsMatrixQueue.push_back(positions_mat);
            PlotPositionsMatrixQueue.push_back(positions_mat);
            PositionsVectorQueue.pop_front();
        }
    }
}

void plot_events(double mag, int Nx, int Ny, const std::string& position_method, double eps, bool enable_tracking, bool enable_event_log, const std::string& event_file, bool report_average, const bool& active) {
    int y_increment = (int)(mag * Ny / 2);
    int x_increment = (int)(y_increment * Nx / Ny);
    int thickness = 2;
    int n_samples = 0;
    int prev_x = 0;
    int prev_y = 0;
    std::ofstream stageFile(event_file + "-stage.csv");
    auto start = std::chrono::high_resolution_clock::now();

    cv::startWindowThread();
    cv::namedWindow("PLOT_EVENTS",
                    cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);

    while (active) {
        while (CVMatrixQueue.empty()) {
            // Do nothing until there is a matrix to process
            if(!active)
                goto stop;
        }
        auto cvmat = CVMatrixQueue.front();
        if (enable_tracking) {
            while (PlotPositionsMatrixQueue.empty()) {
                // Do nothing until there is a corresponding positions vector
                if(!active)
                    goto stop;
            }

            int x_min, x_max, y_min, y_max;
            auto positions = PlotPositionsMatrixQueue.front();
            auto stage_positions = get_position(position_method, positions, previous_positions_plot, eps);

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

                if(enable_event_log)
                    stageFile << timestamp_ms.count() << ","
                              << x_stage << ","
                              << y_stage << "\n";
            }

            cv::putText(cvmat,
                        std::string("Objects: ") + std::to_string((int)(stage_positions.size()/2)), //text
                        cv::Point((int)(0.05*Nx),(int)(0.95*Ny)),
                        cv::FONT_HERSHEY_DUPLEX,
                        0.5,
                        CV_RGB(118, 185, 0),
                        2);

            int first_x = 0;
            int first_y = 0;
            if (stage_positions.n_cols > 0) {
                n_samples += 1;
                first_x = (int) (stage_positions(0, 0) - ((float) Nx / 2));
                first_y = (int) (((float) Ny / 2) - stage_positions(1, 0));
                if (report_average) {
                    first_x = (int)update_average(prev_x, first_x, n_samples);
                    first_y = (int)update_average(prev_y, first_y, n_samples);
                    if (n_samples > 500)
                        n_samples = 0;
                }
            }
            cv::putText(cvmat,
                        std::string("(") + std::to_string(first_x) + std::string(", ") + std::to_string(first_y) + std::string(")"), //text
                        cv::Point((int)(0.80*Nx), (int)(0.95*Ny)),
                        cv::FONT_HERSHEY_DUPLEX,
                        0.5,
                        CV_RGB(118, 185, 0),
                        2);

            prev_x = first_x;
            prev_y = first_y;
            PlotPositionsMatrixQueue.pop_front();
        }
        cv::imshow("PLOT_EVENTS", cvmat);
        cv::waitKey(1);
        CVMatrixQueue.pop_front();
    }
    stop:
    stageFile.close();
    cv::destroyWindow("PLOT_EVENTS");
}

void read_packets(int Nx, int Ny, bool enable_event_log, const std::string& event_file, const bool& active) {
    std::ofstream eventFile(event_file + "-events.csv");
    while (active) {
        while (!PlottingPacketQueue.empty()) {
            if (!active)
                goto stop;
            auto events = PlottingPacketQueue.front();
            cv::Mat cvEvents(Ny, Nx, CV_8UC3, cv::Vec3b{127, 127, 127});
            if (!events.empty()) {
                for (int i = 0; i < events.size(); i += 4) {
                    double ts = events.at(i);
                    int x = (int) events.at(i + 1);
                    int y = (int) events.at(i + 2);
                    int pol = (int) events.at(i + 3);
                    cvEvents.at<cv::Vec3b>(y, x) = pol ? cv::Vec3b{255, 255, 255} : cv::Vec3b{0, 0, 0};
                    if(enable_event_log)
                        eventFile << ts << ","
                                  << x << ","
                                  << y << ","
                                  << pol << "\n";
                }
                if(enable_event_log)
                    eventFile << "\n";
            }
            CVMatrixQueue.push_back(cvEvents);
            PlottingPacketQueue.pop_front();
        }
    }
    stop:
    eventFile.close();
}

void runner(std::thread& reader, std::thread& plotter, std::thread& tracker, std::thread& imager, std::thread& matrix_writer, bool wipe_stage, bool verbose, bool& active) {
    printf("Press space to stop...\n");
    int i = 0;
    while(active) {
        if (key_is_pressed(XK_space)) {
            active = false;
        }
        if (i > 20) {
            i = 0;
            if (verbose) {
                printf("Queue Sizes:\n");
                printf("------------\n");
                printf("Plotting Queue: %zu\n", PlottingPacketQueue.size());
                printf("Event Vector Queue: %zu\n", TrackingVectorQueue.size());
                printf("Image Matrix Queue: %zu\n", CVMatrixQueue.size());
                printf("Matrix Writer Queue: %zu\n", PositionsVectorQueue.size());
                printf("Plot Position Queue: %zu\n", PlotPositionsMatrixQueue.size());
                printf("Stage Position Queue: %zu\n\n", StagePositionsMatrixQueue.size());
            }
        }
        if (wipe_stage) {
            //StagePositionsMatrixQueue.clear();
        }
        i += 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    reader.join();
    plotter.join();
    tracker.join();
    imager.join();
    matrix_writer.join();

    StagePositionsMatrixQueue.clear();
    PlotPositionsMatrixQueue.clear();
    TrackingVectorQueue.clear();
    PositionsVectorQueue.clear();
    PlottingPacketQueue.clear();
    CVMatrixQueue.clear();
}

void launch_threads(const std::string& device_type, double integrationtime, int num_packets, bool enable_tracking, const std::string& position_method, double eps, bool enable_event_log, std::string event_file, double mag, json noise_params, bool wipe_stage, bool report_average, bool verbose, bool& active) {
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

    if (device_type == "xplorer") {
        int Nx = 640;
        int Ny = 480;
        std::thread writing_thread(read_xplorer, num_packets, enable_tracking, noise_params, std::ref(active));
        std::thread plotting_thread(read_packets, Nx, Ny, enable_event_log, event_file, std::ref(active));
        std::thread tracking_thread(tracker, integrationtime, algo, enable_tracking, std::ref(active));
        std::thread matrix_thread(positions_vector_to_matrix, std::ref(active));
        std::thread image_thread(plot_events, mag, Nx, Ny, position_method, eps, enable_tracking, enable_event_log, event_file, report_average, std::ref(active));
        std::thread running_thread(runner, std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(image_thread), std::ref(matrix_thread), wipe_stage, verbose, std::ref(active));
        running_thread.join();
    }
    else {
        int Nx = 346;
        int Ny = 260;
        std::thread writing_thread(read_davis, num_packets, enable_tracking, noise_params, std::ref(active));
        std::thread plotting_thread(read_packets, Nx, Ny, enable_event_log, event_file, std::ref(active));
        std::thread tracking_thread(tracker, integrationtime, algo, enable_tracking, std::ref(active));
        std::thread matrix_thread(positions_vector_to_matrix, std::ref(active));
        std::thread image_thread(plot_events, mag, Nx, Ny, position_method, eps, enable_tracking, enable_event_log, event_file, report_average, std::ref(active));
        std::thread running_thread(runner, std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(image_thread), std::ref(matrix_thread), wipe_stage, verbose, std::ref(active));
        running_thread.join();
    }
}

void drive_stage(const std::string& position_method, double eps, bool enable_stage, const std::string& device_type, double integrationtime, int num_packets, double mag, const json& noise_params, double stage_update, bool& active) {
    if (enable_stage) {
        std::mutex mtx;
        float begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error;
        float prev_pan_position = 0;
        float prev_tilt_position = 0;
        double hfovx, hfovy, y0, r;
        int nx, ny;
        Stage kessler("192.168.50.1", 5520);
        kessler.handshake();
        std::cout << kessler.get_device_info().to_string();
        std::thread pinger(ping, std::ref(kessler), std::ref(mtx), std::ref(active));
        // Temporarily open tracking window to aid in calibration
        bool cal_active = true;
        std::thread cal_thread(launch_threads, device_type, integrationtime, num_packets, true, position_method, eps, false, "~/", mag, noise_params, true, true, false, std::ref(cal_active));
        std::tie(nx, ny, hfovx, hfovy, y0, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error) = calibrate_stage(std::ref(kessler));
        cal_active = false;
        cal_thread.join();
        printf("Enter approximate target distance in meters:\n");
        std::cin >> r;
        prepareStage.release();
        auto start = std::chrono::high_resolution_clock::now();
        while (active) {
            if (!StagePositionsMatrixQueue.empty()) {
                if (!active)
                    goto stop_stage;
                auto positions = StagePositionsMatrixQueue.front();
                if (positions.n_cols > 0) { // Stay in place if no object found
                    auto stage_positions = get_position(position_method, positions, previous_positions_stage, eps);

                    // Go to first position in list. Selecting between objects to be implemented later.
                    double x = stage_positions(0,0) - ((double) nx / 2);
                    double y = ((double) ny / 2) - stage_positions(1,0);

                    double theta = get_theta(y, ny, hfovy);
                    double phi = get_phi(x, nx, hfovx);
                    double theta_prime = get_theta_prime(phi, theta, y0, r, theta_prime_error);
                    double phi_prime = get_phi_prime(phi, theta, y0, r, phi_prime_error);

                    float pan_position = get_pan_position(begin_pan, end_pan, phi_prime);
                    float tilt_position = get_tilt_position(begin_tilt, end_tilt, theta_prime);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> since_last = end - start;

                    bool move = move_stage(pan_position, prev_pan_position, tilt_position, prev_tilt_position, stage_update);

                    if (since_last.count() > 50 && move) {
                        printf("Calculated Stage Angles: (%0.2f, %0.2f)\n", theta_prime * 180 / PI,
                               phi_prime * 180 / PI);
                        printf("Stage Positions:\n     Pan: %0.2f (End: %0.2f)\n     Tilt: %0.2f (End: %0.2f)\n",
                               pan_position, end_pan - begin_pan, tilt_position, end_tilt - begin_tilt);
                        printf("Moving stage to (%.2f, %.2f)\n\n", x, y);

                        mtx.lock();
                        kessler.set_position_speed_acceleration(2, pan_position, (float)0.6*PAN_MAX_SPEED, PAN_MAX_ACC);
                        kessler.set_position_speed_acceleration(3, tilt_position, (float)0.6*TILT_MAX_SPEED, TILT_MAX_ACC);
                        mtx.unlock();

                        prev_pan_position = pan_position;
                        prev_tilt_position = tilt_position;

                        start = std::chrono::high_resolution_clock::now();
                    }
                }
                StagePositionsMatrixQueue.pop_front();
            }
        }
        stop_stage:
        pinger.join();
    }
    else {
        prepareStage.release();
        while (active) {
            if (!StagePositionsMatrixQueue.empty()) {
                StagePositionsMatrixQueue.pop_front();
            }
        }
    }
}