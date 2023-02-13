# define eigen_assert(X) do { if(!(X)) throw std::runtime_error(#X); } while(false);

#include <libcaercpp/devices/dvxplorer.hpp>
#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/filters/dvs_noise.hpp>

#include <atomic>
#include <csignal>
#include <tuple>
#include <queue>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ebs-tracking/Algorithm.hpp"

#include <X11/Xlib.h>
#include <X11/keysym.h>

static std::atomic_bool globalShutdown(false);

std::queue<std::vector<double>> PlottingPacketQueue;
std::queue<std::vector<double>> TrackingVectorQueue;
std::queue<cv::Mat> CVMatrixQueue;
std::queue<std::vector<double>> PositionsVectorQueue;

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

// Read csv and process in "real time"
// dt: integration time in ms
// delay: time behind actual in ms
void tracker (double dt, DBSCAN_KNN T, bool enable_tracking, bool&active) {
    if (enable_tracking) {
        while (active) {
            while (!TrackingVectorQueue.empty() && active) {
                auto events = TrackingVectorQueue.front();
                std::vector<double> positions;
                try {
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
                            T(mem, N);
                            Eigen::MatrixXd targets{T.currentTracks()};

                            for (int i{0}; i < targets.rows(); ++i) {
                                positions.push_back(targets(i, 0));
                                positions.push_back(targets(i, 1));
                            }

                            // Break once all events have been used.
                            if (t0 > events[4 * (nEvents - 1)])
                                break;

                            // Evolve tracks in time.
                            T.predict();

                            // Update eventIdx
                            mem += 4 * N;
                        }
                        PositionsVectorQueue.push(positions);
                        TrackingVectorQueue.pop();
                    } else {
                        positions.push_back(-10);
                        positions.push_back(-10);
                        PositionsVectorQueue.push(positions);
                        TrackingVectorQueue.pop();
                    }
                }
                catch (...) {
                    positions.push_back(-10);
                    positions.push_back(-10);
                    PositionsVectorQueue.push(positions);
                    TrackingVectorQueue.pop();
                }
            }
        }
    }
}

void plot_events(double mag, int Nx, int Ny, bool enable_tracking, bool& active) {
    int y_increment = (int)(mag * Ny / 2);
    int x_increment = (int)(y_increment * Nx / Ny);

    cv::namedWindow("PLOT_EVENTS",
                   cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);
    while (active) {
        while (CVMatrixQueue.empty() && active) {
            // Do nothing until there is a matrix to process
        }
        auto cvmat = CVMatrixQueue.front();
        if (enable_tracking) {
            while (PositionsVectorQueue.empty() && active) {
                // Do nothing until there is a corresponding positions vector
            }

            auto positions = PositionsVectorQueue.front();

            for (int i=0; i < positions.size(); i += 2) {
                int x = (int)positions[i];
                int y = (int)positions[i+1];
                if (x == -10 || y == -10) {
                    continue;
                }

                int y_min = std::max(y - y_increment, 0);
                int x_min = std::max(x - x_increment, 0);
                int y_max = std::min(y + y_increment, Ny - 1);
                int x_max = std::min(x + x_increment, Nx - 1);

                cv::Point p1(x_min, y_min);
                cv::Point p2(x_max, y_max);
                int thickness = 2;
                rectangle(cvmat, p1, p2,
                          cv::Scalar(255, 0, 0),
                          thickness, cv::LINE_8);


            }

            PositionsVectorQueue.pop();
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

std::tuple<double, double> process_packet(double time_start, double time_current, double aggregationtime, bool enable_tracking, const libcaer::filters::DVSNoise& dvsNoiseFilter, std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer) {
    std::vector<double> events;
    if (packetContainer == nullptr) {
        std::tuple<double, double> ret = {time_start, time_current};
        return ret;
    }

    for (auto &packet: *packetContainer) {
        if (packet == nullptr) {
            //printf("Packet is empty (not present).\n");
            continue; // Skip if nothing there.
        }

        if (packet->getEventType() == POLARITY_EVENT) {
            std::shared_ptr<libcaer::events::PolarityEventPacket> polarity
                    = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

            const libcaer::events::PolarityEvent &firstEvent = (*polarity)[0];
            double ts = (double)firstEvent.getTimestamp()/1000;
            if (time_start == 0 && time_current == 0) {
                time_start = ts;
            }
            if (ts - time_start < aggregationtime) {
                time_current = ts;
            }
            else {
                time_start = ts;
                time_current = ts;
            }

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
            if (enable_tracking) {
                TrackingVectorQueue.push(events);
            }
            PlottingPacketQueue.push(events);
        }
    }
    std::tuple<double, double> ret = {time_start, time_current};
    return ret;
}

int read_xplorer (double dt, double aggregationtime, bool enable_tracking, bool& active) {
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

    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        double time_start = 0;
        double time_current = 0;
        while (time_current - time_start < aggregationtime) {
            std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
            std::tie(time_start, time_current) = process_packet(time_start, time_current, dt, enable_tracking, dvsNoiseFilter, std::move(packetContainer));
        }
    }
    handle.dataStop();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

int read_davis (double dt, double aggregationtime, bool enable_tracking, bool& active) {
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

    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        double time_start = 0;
        double time_current = 0;
        while (time_current - time_start < aggregationtime) {
            std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = davisHandle.dataGet();
            std::tie(time_start, time_current) = process_packet(time_start, time_current, dt, enable_tracking, dvsNoiseFilter, std::move(packetContainer));
        }
    }
    davisHandle.dataStop();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

bool key_is_pressed(KeySym ks) {
    Display *dpy = XOpenDisplay(":0");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    bool isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
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
            printf("Position Vector Queue: %zu\n\n\n", PositionsVectorQueue.size());
        }
        i += 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    reader.join();
    plotter.join();
    tracker.join();
    imager.join();
}

int main(int argc, char* argv[]) {
    /*
     Simulate live data tracking using a CSV file containing event data.
     A separate Matlab script can be used to generate GIFs of the result.

     Args:
          argv[1]: Device type: "xplorer" or "davis"
          argv[2]: Integration time in milliseconds.
          argv[3]: Aggregation time in milliseconds.
          argv[4]: Tracking: 0 for disabled, 1 for enabled
          argv[5]: Magnification

     Ret:
          0
     */

    if (argc != 6) {
        printf("Invalid number of arguments.\n");
        return 0;
    }

    std::string device_type = {std::string(argv[1])};
    double integrationtime = {std::stod(argv[2])};
    double aggregationtime = {std::stod(argv[3])};
    bool enable_tracking = {std::stoi(argv[4])!=0};
    double mag = {std::stod(argv[5])};

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


    bool active = true;

    if (device_type == "xplorer") {
        int Nx = 640;
        int Ny = 480;
        std::thread writing_thread(read_xplorer, integrationtime, aggregationtime, enable_tracking, std::ref(active));
        std::thread plotting_thread(read_packets, Nx, Ny, std::ref(active));
        std::thread tracking_thread(tracker, integrationtime, algo, enable_tracking, std::ref(active));
        std::thread image_thread(plot_events, mag, Nx, Ny, enable_tracking, std::ref(active));
        std::thread running_thread(runner, std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(image_thread), std::ref(active));
        running_thread.join();
    }
    else {
        int Nx = 346;
        int Ny = 260;
        std::thread writing_thread(read_davis, integrationtime, aggregationtime, enable_tracking, std::ref(active));
        std::thread plotting_thread(read_packets, Nx, Ny, std::ref(active));
        std::thread tracking_thread(tracker, integrationtime, algo, enable_tracking, std::ref(active));
        std::thread image_thread(plot_events, mag, Nx, Ny, enable_tracking, std::ref(active));
        std::thread running_thread(runner, std::ref(writing_thread), std::ref(plotting_thread), std::ref(tracking_thread), std::ref(image_thread), std::ref(active));
        running_thread.join();
    }

    return 0;
}