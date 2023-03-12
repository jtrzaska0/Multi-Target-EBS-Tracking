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

class Buffers {
    public:
        boost::circular_buffer<std::vector<double>> PacketQueue;
        arma::mat prev_positions;
        Buffers(const int buffer_size, const int history_size){
            PacketQueue.set_capacity(buffer_size);
            prev_positions.set_size(2, history_size);
            prev_positions.fill(arma::fill::zeros);
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

int read_xplorer (Buffers& buffers, const json& noise_params, bool verbose, bool enable_filter, bool& active) {
    // Install signal handler for global shutdown.
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

    printf("Press space to stop...\n");
    auto start = std::chrono::high_resolution_clock::now();
    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        std::vector<double> events;
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
        if (packetContainer == nullptr || buffers.PacketQueue.full()) {
            continue;
        }
        for (auto &packet: *packetContainer) {
            if (packet == nullptr) {
                continue; // Skip if nothing there.
            }

            if (packet->getEventType() == POLARITY_EVENT) {
                std::shared_ptr<libcaer::events::PolarityEventPacket> polarity
                        = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                if(enable_filter)
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
            }
        }
        buffers.PacketQueue.push_back(events);
        if (key_is_pressed(XK_space)) {
            active = false;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> timestamp_ms = end - start;
        if (timestamp_ms.count() > 2000 && verbose) {
            start = std::chrono::high_resolution_clock::now();
            printf("Packet Queue: %zu\n", buffers.PacketQueue.size());
        }
    }
    handle.dataStop();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

int read_davis (Buffers& buffers, const json& noise_params, bool verbose, bool enable_filter, bool& active) {
    // Install signal handler for global shutdown.
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
    if (davis_info.dvsHasBackgroundActivityFilter && enable_filter) {
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

    printf("Press space to stop...\n");
    auto start = std::chrono::high_resolution_clock::now();
    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        std::vector<double> events;
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = davisHandle.dataGet();
        if (packetContainer == nullptr || buffers.PacketQueue.full()) {
            continue;
        }

        for (auto &packet: *packetContainer) {
            if (packet == nullptr) {
                continue; // Skip if nothing there.
            }

            if (packet->getEventType() == POLARITY_EVENT) {
                std::shared_ptr<libcaer::events::PolarityEventPacket> polarity
                        = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                if (enable_filter)
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
            }
        }
        buffers.PacketQueue.push_back(events);
        if (key_is_pressed(XK_space)) {
            active = false;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> timestamp_ms = end - start;
        if (timestamp_ms.count() > 2000 && verbose) {
            start = std::chrono::high_resolution_clock::now();
            printf("Packet Queue: %zu\n", buffers.PacketQueue.size());
        }
    }
    davisHandle.dataStop();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

void processing_threads(Buffers& buffers, Stage* kessler, double dt, DBSCAN_KNN T, bool enable_tracking,
                       int Nx, int Ny, bool enable_event_log, const std::string& event_file,
                       double mag, const std::string& position_method, double eps, const bool& active,
                       std::tuple<int, int, double, double, double, float, float, float, float, float, float, double> cal_params) {
    auto const[nx, ny, hfovx, hfovy, y0, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error, phi_prime_error, r] = cal_params;
    auto start = std::chrono::high_resolution_clock::now();
    std::ofstream stageFile(event_file + "-stage.csv");
    std::ofstream eventFile(event_file + "-events.csv");
    while (active) {
        bool A_processed = false;
        bool B_processed = false;
        if (buffers.PacketQueue.empty())
            continue;
        std::future<std::tuple<cv::Mat, arma::mat, std::string, std::string>> fut_resultA =
                std::async(std::launch::async, process_packet, buffers.PacketQueue.front(), dt, T, enable_tracking, Nx,
                           Ny, enable_event_log, buffers.prev_positions, mag, position_method, eps);
        buffers.PacketQueue.pop_front();

        fill_processorB:
        if (!active)
            continue;
        if (buffers.PacketQueue.empty()) {
            if (!A_processed && fut_resultA.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                A_processed = true;
                start = read_future(fut_resultA, stageFile, eventFile, kessler, nx, ny, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error,
                                   phi_prime_error, hfovx, hfovy, y0, r, start);
            }
            goto fill_processorB;
        }
        std::future<std::tuple<cv::Mat, arma::mat, std::string, std::string>> fut_resultB =
                std::async(std::launch::async, process_packet, buffers.PacketQueue.front(), dt, T, enable_tracking, Nx,
                           Ny, enable_event_log, buffers.prev_positions, mag, position_method, eps);
        buffers.PacketQueue.pop_front();

        fill_processorC:
        if (!active)
            continue;
        if (buffers.PacketQueue.empty()) {
            if (!A_processed && fut_resultA.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                A_processed = true;
                start = read_future(fut_resultA, stageFile, eventFile, kessler, nx, ny, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error,
                                    phi_prime_error, hfovx, hfovy, y0, r, start);
            }
            if (!B_processed && fut_resultB.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                B_processed = true;
                start = read_future(fut_resultB, stageFile, eventFile, kessler, nx, ny, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error,
                                    phi_prime_error, hfovx, hfovy, y0, r, start);
            }
            goto fill_processorC;
        }
        std::future<std::tuple<cv::Mat, arma::mat, std::string, std::string>> fut_resultC =
                std::async(std::launch::async, process_packet, buffers.PacketQueue.front(), dt, T, enable_tracking, Nx,
                           Ny, enable_event_log, buffers.prev_positions, mag, position_method, eps);
        buffers.PacketQueue.pop_front();

        if (!A_processed) {
            start = read_future(fut_resultA, stageFile, eventFile, kessler, nx, ny, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error,
                                phi_prime_error, hfovx, hfovy, y0, r, start);
        }
        if (!B_processed) {
            start = read_future(fut_resultB, stageFile, eventFile, kessler, nx, ny, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error,
                                phi_prime_error, hfovx, hfovy, y0, r, start);
        }
        start = read_future(fut_resultC, stageFile, eventFile, kessler, nx, ny, begin_pan, end_pan, begin_tilt, end_tilt, theta_prime_error,
                            phi_prime_error, hfovx, hfovy, y0, r, start);
    }
    stageFile.close();
    eventFile.close();
}