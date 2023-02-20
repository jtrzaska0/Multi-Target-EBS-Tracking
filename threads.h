#pragma once

#include <atomic>
#include <csignal>
#include <queue>
#include <semaphore>

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

static std::atomic_bool globalShutdown(false);

std::queue<std::vector<double>> PlottingPacketQueue;
std::queue<std::vector<double>> TrackingVectorQueue;
std::queue<cv::Mat> CVMatrixQueue;
std::queue<std::vector<double>> PlotPositionsVectorQueue;
std::queue<std::vector<double>> StagePositionsVectorQueue;
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

int read_xplorer (int num_packets, bool enable_tracking, bool& active) {
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

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TWO_LEVELS, true);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_CHECK_POLARITY, true);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MIN, 2);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MAX, 8);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME, 2000);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE, true);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME, 200);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE, true);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_ENABLE, true);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_LEARN, true);

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
                        TrackingVectorQueue.push(events);
                    }
                    PlottingPacketQueue.push(events);
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

int read_davis (int num_packets, bool enable_tracking, bool& active) {
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
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_BACKGROUND_ACTIVITY_TIME, 8);
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_BACKGROUND_ACTIVITY, true);

        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_REFRACTORY_PERIOD_TIME, 1);
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_REFRACTORY_PERIOD, true);
    }

    // Add full-sized software filter to reduce DVS noise.
    libcaer::filters::DVSNoise dvsNoiseFilter = libcaer::filters::DVSNoise(davis_info.dvsSizeX, davis_info.dvsSizeY);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TWO_LEVELS, true);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_CHECK_POLARITY, true);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MIN, 2);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MAX, 8);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME, 2000);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE, true);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME, 200);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE, true);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_ENABLE, true);
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_LEARN, true);

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
                        TrackingVectorQueue.push(events);
                    }
                    PlottingPacketQueue.push(events);
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

void tracker (double dt, DBSCAN_KNN T, bool enable_tracking, bool&active) {
    if (enable_tracking) {
        while (active) {
            while (!TrackingVectorQueue.empty() && active) {
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
                PlotPositionsVectorQueue.push(positions);
                StagePositionsVectorQueue.push(positions);
                TrackingVectorQueue.pop();
            }
        }
    }
}

void plot_events(double mag, int Nx, int Ny, const std::string& position_method, double eps, bool enable_tracking, bool& active) {
    int y_increment = (int)(mag * Ny / 2);
    int x_increment = (int)(y_increment * Nx / Ny);
    int thickness = 2;

    cv::namedWindow("PLOT_EVENTS",
                    cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);
    while (active) {
        while (CVMatrixQueue.empty() && active) {
            // Do nothing until there is a matrix to process
        }
        auto cvmat = CVMatrixQueue.front();
        if (enable_tracking) {
            while (PlotPositionsVectorQueue.empty() && active) {
                // Do nothing until there is a corresponding positions vector
            }

            int x_min, x_max, y_min, y_max;
            auto positions = PlotPositionsVectorQueue.front();
            auto stage_positions = get_position(position_method, positions, eps);

            for (int i=0; i < positions.size(); i += 2) {
                int x = (int)positions[i];
                int y = (int)positions[i+1];
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

            for (int i = 0; i < stage_positions.size(); i += 2) {
                int x_stage = (int)stage_positions[i];
                int y_stage = (int)stage_positions[i+1];
                y_min = std::max(y_stage - y_increment, 0);
                x_min = std::max(x_stage - x_increment, 0);
                y_max = std::min(y_stage + y_increment, Ny - 1);
                x_max = std::min(x_stage + x_increment, Nx - 1);

                cv::Point p1_stage(x_min, y_min);
                cv::Point p2_stage(x_max, y_max);

                rectangle(cvmat, p1_stage, p2_stage,
                          cv::Scalar(0, 0, 255),
                          thickness, cv::LINE_8);
            }

            PlotPositionsVectorQueue.pop();
        }


        cv::imshow("PLOT_EVENTS", cvmat);
        cv::waitKey(1);

        CVMatrixQueue.pop();
    }
    cv::destroyWindow("PLOT_EVENTS");
}

void read_packets(int Nx, int Ny, bool& active) {
    while (active) {
        while (!PlottingPacketQueue.empty() && active) {
            auto events = PlottingPacketQueue.front();
            cv::Mat cvEvents(Ny, Nx, CV_8UC3, cv::Vec3b{127, 127, 127});
            if (!events.empty()) {
                for (int i = 0; i < events.size(); i += 4) {
                    int x = (int) events.at(i + 1);
                    int y = (int) events.at(i + 2);
                    int pol = (int) events.at(i + 3);
                    cvEvents.at<cv::Vec3b>(y, x) = pol ? cv::Vec3b{255, 255, 255} : cv::Vec3b{0, 0, 0};
                }
            }
            CVMatrixQueue.push(cvEvents);
            PlottingPacketQueue.pop();
        }
    }
}

void runner(std::thread& reader, std::thread& plotter, std::thread& tracker, std::thread& imager, bool& active) {
    printf("Press space to stop...\n");
    int i = 0;
    while(active) {
        if (key_is_pressed(XK_space)) {
            active = false;
        }
        if (i > 20) {
            i = 0;
            printf("Queue Sizes:\n");
            printf("------------\n");
            printf("Plotting Queue: %zu\n", PlottingPacketQueue.size());
            printf("Event Vector Queue: %zu\n", TrackingVectorQueue.size());
            printf("Image Matrix Queue: %zu\n", CVMatrixQueue.size());
            printf("Plot Position Queue: %zu\n", PlotPositionsVectorQueue.size());
            printf("Stage Position Queue: %zu\n\n", StagePositionsVectorQueue.size());
        }
        i += 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    reader.join();
    plotter.join();
    tracker.join();
    imager.join();
}

void launch_threads(const std::string& device_type, double integrationtime, int num_packets, bool enable_tracking, const std::string& position_method, double eps, double mag, bool& active) {
    /**Create an Algorithm object here.**/
    // Matrix initializer
    // DBSCAN
    Eigen::MatrixXd invals {Eigen::MatrixXd::Zero(1, 4)};
/*     invals(0, 0) = 8;
     invals(0, 1) = 8;
     invals(0, 2) = 1.2;
*/
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
        std::thread writing_thread(read_xplorer, num_packets, enable_tracking, std::ref(active));
        std::thread plotting_thread(read_packets, Nx, Ny, std::ref(active));
        std::thread tracking_thread(tracker, integrationtime, algo, enable_tracking, std::ref(active));
        std::thread image_thread(plot_events, mag, Nx, Ny, position_method, eps, enable_tracking, std::ref(active));
        std::thread running_thread(runner, std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(image_thread), std::ref(active));
        running_thread.join();
    }
    else {
        int Nx = 346;
        int Ny = 260;
        std::thread writing_thread(read_davis, num_packets, enable_tracking, std::ref(active));
        std::thread plotting_thread(read_packets, Nx, Ny, std::ref(active));
        std::thread tracking_thread(tracker, integrationtime, algo, enable_tracking, std::ref(active));
        std::thread image_thread(plot_events, mag, Nx, Ny, position_method, eps, enable_tracking, std::ref(active));
        std::thread running_thread(runner, std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(image_thread), std::ref(active));
        running_thread.join();
    }
}

void drive_stage(const std::string& position_method, double eps, bool enable_stage, bool& active) {
    if (enable_stage) {
        std::mutex mtx;
        float begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error;
        double hfovx, hfovy, y0, r;
        int nx, ny;
        Stage kessler("192.168.50.1", 5520);
        kessler.handshake();
        std::cout << kessler.get_device_info().to_string();
        std::thread pinger(ping, std::ref(kessler), std::ref(mtx), std::ref(active));
        std::tie(nx, ny, hfovx, hfovy, y0, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error) = calibrate_stage(std::ref(kessler));
        printf("Enter approximate target distance in meters:\n");
        std::cin >> r;
        prepareStage.release();
        while (active) {
            if (!StagePositionsVectorQueue.empty()) {
                std::vector<double> positions = StagePositionsVectorQueue.front();
                if (!positions.empty()) { // Stay in place if no object found
                    std::vector<double> stage_positions = get_position(position_method, positions, eps);

                    // Go to first position in list. Selecting between objects to be implemented later.
                    double x = stage_positions[0] - ((double) nx / 2);
                    double y = ((double) ny / 2) - stage_positions[1];

                    double theta = get_theta(y, ny, hfovy);
                    double phi = get_phi(x, nx, hfovx);
                    double theta_prime = get_theta_prime(phi, theta, y0, r, theta_prime_error);
                    double phi_prime = get_phi_prime(phi, theta, y0, r, phi_prime_error);

                    float pan_position = get_pan_position(begin_pan, end_pan, phi_prime);
                    float tilt_position = get_tilt_position(begin_tilt, end_tilt, theta_prime);
                    printf("Moving stage to (%.2f, %.2f)\n", x, y);

                    mtx.lock();
                    kessler.set_position_speed_acceleration(2, pan_position, PAN_MAX_SPEED, PAN_MAX_ACC);
                    kessler.set_position_speed_acceleration(3, tilt_position, TILT_MAX_SPEED, TILT_MAX_ACC);
                    mtx.unlock();
                }
                StagePositionsVectorQueue.pop();
            }
        }
        pinger.join();
    }
    else {
        prepareStage.release();
        while (active) {
            if (!StagePositionsVectorQueue.empty()) {
                StagePositionsVectorQueue.pop();
            }
        }
    }
}