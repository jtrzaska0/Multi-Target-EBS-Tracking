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
std::counting_semaphore<1> prepareStage(0);
std::counting_semaphore<1> sema_trackerProcessNext(1);
std::counting_semaphore<1> sema_trackerFinished(0);
std::counting_semaphore<1> sema_trackerFinishedPlotter(0);
std::counting_semaphore<1> sema_plotterProcessNext(1);
std::counting_semaphore<1> sema_plotterFinished(0);
std::counting_semaphore<1> sema_trackerFinishedStage(0);
std::counting_semaphore<1> sema_stageFinished(0);
std::counting_semaphore<1> sema_stageProcessNext(1);

class Buffers {
    public:
        boost::circular_buffer<std::vector<double>> PacketQueue;
        arma::mat previous_positions;
        arma::mat positions;
        Buffers(const int buffer_size, const int history_size){
            PacketQueue.set_capacity(buffer_size);
            previous_positions.set_size(2, history_size);
            previous_positions.fill(arma::fill::zeros);
        }
};

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

int read_xplorer (Buffers& buffers, int num_packets, const json& noise_params, const bool& active) {
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
                    buffers.PacketQueue.push_back(events);
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

int read_davis (Buffers& buffers, int num_packets, const json& noise_params, const bool& active) {
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
                    buffers.PacketQueue.push_back(events);
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

void tracker (Buffers& buffers, double dt, const DBSCAN_KNN& T, bool enable_tracking, const bool& active) {
    while (active) {
        if (!sema_trackerProcessNext.try_acquire())
            continue;
        if (buffers.PacketQueue.empty()) {
            sema_trackerProcessNext.release();
            continue;
        }
        auto events = buffers.PacketQueue.front();
        buffers.positions = run_tracker(events, dt, T, enable_tracking);
        sema_trackerFinishedPlotter.release();
        sema_trackerFinishedStage.release();
        sema_trackerFinished.release();
    }
}

void plotter (Buffers& buffers, double mag, int Nx, int Ny, const std::string& position_method, double eps, bool enable_tracking, bool enable_event_log, const std::string& event_file, const bool& active) {
    cv::startWindowThread();
    cv::namedWindow("PLOT_EVENTS",
                    cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);
    while (active) {
        if (!sema_plotterProcessNext.try_acquire())
            continue;
        if (buffers.PacketQueue.empty()) {
            sema_plotterProcessNext.release();
            continue;
        }
        if (!sema_trackerFinishedPlotter.try_acquire()) {
            sema_plotterProcessNext.release();
            continue;
        }
        auto events = buffers.PacketQueue.front();
        cv::Mat cvMat = read_packets(events, Nx, Ny, enable_event_log, event_file);
        update_window(cvMat, buffers.positions, buffers.previous_positions, mag, Nx, Ny, position_method, eps, enable_tracking, enable_event_log, event_file);
        sema_plotterFinished.release();
    }
    cv::destroyWindow("PLOT_EVENTS");
}

void clear_packets (Buffers& buffers, const bool& active) {
    while (active) {
        if (!sema_trackerFinished.try_acquire())
            continue;
        if (!sema_plotterFinished.try_acquire()) {
            sema_trackerFinished.release();
            continue;
        }
        if (!sema_stageFinished.try_acquire()) {
            sema_trackerFinished.release();
            sema_plotterFinished.release();
            continue;
        }
        buffers.PacketQueue.pop_front();
        sema_trackerProcessNext.release();
        sema_plotterProcessNext.release();
        sema_stageProcessNext.release();
    }
}

void runner(Buffers& buffers, std::thread& reader, std::thread& plotter, std::thread& tracker, std::thread& packet_clearer, bool verbose, bool& active) {
    printf("Press space to stop...\n");
    int i = 0;
    while(active) {
        if (key_is_pressed(XK_space)) {
            active = false;
        }
        if (i > 20) {
            i = 0;
            if (verbose) {
                printf("Packet Queue: %zu\n", buffers.PacketQueue.size());
            }
        }
        i += 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    reader.join();
    plotter.join();
    tracker.join();
    packet_clearer.join();
}

void launch_threads(Buffers& buffers, const std::string& device_type, double integrationtime, int num_packets, bool enable_tracking, const std::string& position_method, double eps, bool enable_event_log, std::string event_file, double mag, json noise_params, bool report_average, bool verbose, bool& active) {
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
        std::thread packet_clearer(clear_packets, std::ref(buffers), std::ref(active));
        std::thread writing_thread(read_xplorer, std::ref(buffers), num_packets, noise_params, std::ref(active));
        std::thread plotting_thread(plotter, std::ref(buffers), mag, Nx, Ny, position_method, eps, enable_tracking, enable_event_log, event_file, std::ref(active));
        std::thread tracking_thread(tracker, std::ref(buffers), integrationtime, algo, enable_tracking, std::ref(active));
        std::thread running_thread(runner, std::ref(buffers), std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(packet_clearer), verbose, std::ref(active));
        running_thread.join();
    }
    else {
        int Nx = 346;
        int Ny = 260;
        std::thread packet_clearer(clear_packets, std::ref(buffers), std::ref(active));
        std::thread writing_thread(read_davis, std::ref(buffers), num_packets, noise_params, std::ref(active));
        std::thread plotting_thread(plotter, std::ref(buffers), mag, Nx, Ny, position_method, eps, enable_tracking, enable_event_log, event_file, std::ref(active));
        std::thread tracking_thread(tracker, std::ref(buffers), integrationtime, algo, enable_tracking, std::ref(active));
        std::thread running_thread(runner, std::ref(buffers), std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(packet_clearer), verbose, std::ref(active));
        running_thread.join();
    }
}

void drive_stage(Buffers& buffers, const std::string& position_method, double eps, bool enable_stage, double stage_update, bool& active) {
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
        std::tie(nx, ny, hfovx, hfovy, y0, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error) = calibrate_stage(std::ref(kessler));
        printf("Enter approximate target distance in meters:\n");
        std::cin >> r;
        prepareStage.release();
        auto start = std::chrono::high_resolution_clock::now();
        while (active) {
            if (!sema_stageProcessNext.try_acquire())
                continue;
            if (!sema_trackerFinishedStage.try_acquire()) {
                sema_stageProcessNext.release();
                continue;
            }
            auto positions = buffers.positions;
            if (positions.n_cols > 0) { // Stay in place if no object found
                auto stage_positions = get_position(position_method, positions, buffers.previous_positions, eps);

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
            sema_stageFinished.release();
        }
        pinger.join();
    }
    else {
        prepareStage.release();
        while (active) {
            if (!sema_stageProcessNext.try_acquire())
                continue;
            if (!sema_trackerFinishedStage.try_acquire()) {
                sema_stageProcessNext.release();
                continue;
            }
            sema_stageFinished.release();
        }
    }
}