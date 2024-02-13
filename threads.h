// File     threads.h
// Summary  Processing threads for EBS and FBS.
// Authors  Trevor Schlack - Modified by Jacob Trzaska
# pragma once

// Standard imports
# include <atomic>
# include <chrono>
# include <csignal>
# include <queue>
# include <semaphore>
# include <boost/lockfree/spsc_queue.hpp>
# include <nlohmann/json.hpp>
# include <libcaercpp/devices/dvxplorer.hpp>
# include <libcaercpp/devices/davis.hpp>
# include <libcaercpp/filters/dvs_noise.hpp>
# include <opencv2/core.hpp>
# include <opencv2/highgui/highgui.hpp>
# include <opencv2/imgproc.hpp>
# include <opencv2/tracking.hpp>
# include <opencv2/dnn.hpp>

// Local imports
# include "pointing.h"
# include "utils.h"
# include "controller.h"
# include "videos.h"

// Namespacing
using json = nlohmann::json;
static std::atomic_bool globalShutdown(false);


class Buffers {
    public:
    boost::lockfree::spsc_queue<std::vector<double>> PacketQueue{1024};
    arma::mat prev_positions;

    explicit Buffers(const int history_size) {
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


int read_xplorer(Buffers &buffers, const bool debug, const json &noise_params, bool enable_filter, const std::string& file,
                 std::chrono::time_point<std::chrono::high_resolution_clock> start, bool &active) {
    // Install signal handler for global shutdown.
    struct sigaction shutdownAction{};
    std::ofstream rateFile(file + "-rates.csv");

    shutdownAction.sa_handler = &globalShutdownSignalHandler;
    shutdownAction.sa_flags = 0;
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
    libcaer::filters::DVSNoise dvsNoiseFilter = libcaer::filters::DVSNoise(xplorer_info.dvsSizeX,
                                                                           xplorer_info.dvsSizeY);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TWO_LEVELS,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_TWO_LEVELS", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_CHECK_POLARITY,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_CHECK_POLARITY", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MIN,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MIN", 2));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MAX,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MAX", 8));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_TIME", 2000));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_ENABLE", true));

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME,
                             noise_params.value("CAER_REFRACTORY_PERIOD_TIME", 200));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE,
                             noise_params.value("CAER_REFRACTORY_PERIOD_ENABLE", true));

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_ENABLE, noise_params.value("CAER_HOTPIXEL_ENABLE", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_LEARN, noise_params.value("CAER_HOTPIXEL_LEARN", true));

    // Now let's get start getting some data from the device. We just loop in blocking mode,
    // no notification needed regarding new events. The shutdown notification, for example if
    // the device is disconnected, should be listened to.
    handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);

    // Let's turn on blocking data-get mode to avoid wasting resources.
    handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

    printf("Press space to stop...\n");
    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        if (debug)
            printf("Started EBS acquisition.\n");
        std::vector<double> events;
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = handle.dataGet();
        if (packetContainer == nullptr) {
            continue;
        }

        auto start_processing = std::chrono::high_resolution_clock::now();
        int eventCount = 0;
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
                    eventCount += 1;
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

        auto stop_processing = std::chrono::high_resolution_clock::now();
        auto processing_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_processing - start_processing);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_processing - start);
        double eventRate = 1000000 * eventCount / (double)processing_duration.count();

        rateFile << (double)total_duration.count() << "," << eventRate << "\n";
        buffers.PacketQueue.push(events);

        //if (key_is_pressed(XK_space)) {
        //    active = false;
        //}

        if (debug)
            printf("Completed EBS acquisition.\n");
    }

    handle.dataStop();
    rateFile.close();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}

int read_davis(Buffers &buffers, const bool debug, const json &noise_params, bool enable_filter, const std::string& file,
               std::chrono::time_point<std::chrono::high_resolution_clock> start, bool &active) {
    // Install signal handler for global shutdown.
    struct sigaction shutdownAction{};
    std::ofstream rateFile(file + "-rates.csv");

    shutdownAction.sa_handler = &globalShutdownSignalHandler;
    shutdownAction.sa_flags = 0;
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
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_BACKGROUND_ACTIVITY_TIME,
                              noise_params.value("DAVIS_BACKGROUND_ACTIVITY_TIME", 8));
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_BACKGROUND_ACTIVITY,
                              noise_params.value("DAVIS_BACKGROUND_ACTIVITY", true));

        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_REFRACTORY_PERIOD_TIME,
                              noise_params.value("DAVIS_REFRACTORY_PERIOD_TIME", 1));
        davisHandle.configSet(DAVIS_CONFIG_DVS, DAVIS_CONFIG_DVS_FILTER_REFRACTORY_PERIOD,
                              noise_params.value("DAVIS_REFRACTORY_PERIOD", true));
    }

    // Add full-sized software filter to reduce DVS noise.
    libcaer::filters::DVSNoise dvsNoiseFilter = libcaer::filters::DVSNoise(davis_info.dvsSizeX, davis_info.dvsSizeY);

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TWO_LEVELS,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_TWO_LEVELS", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_CHECK_POLARITY,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_CHECK_POLARITY", true));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MIN,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MIN", 2));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MAX,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_SUPPORT_MAX", 8));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_TIME", 2000));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE,
                             noise_params.value("CAER_BACKGROUND_ACTIVITY_ENABLE", true));

    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME,
                             noise_params.value("CAER_REFRACTORY_PERIOD_TIME", 200));
    dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE,
                             noise_params.value("CAER_REFRACTORY_PERIOD_ENABLE", true));

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
    while (!globalShutdown.load(std::memory_order_relaxed) && active) {
        if (debug)
            printf("Started EBS acquisition.\n");

        std::vector<double> events;
        std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = davisHandle.dataGet();

        if (packetContainer == nullptr) {
            continue;
        }

        auto start_processing = std::chrono::high_resolution_clock::now();
        int eventCount = 0;

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
                    eventCount += 1;
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

        auto stop_processing = std::chrono::high_resolution_clock::now();
        auto processing_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_processing - start_processing);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_processing - start);
        double eventRate = 1000000 * eventCount / (double)processing_duration.count();

        rateFile << (double)total_duration.count() << "," << eventRate << "\n";
        buffers.PacketQueue.push(events);

        //if (key_is_pressed(XK_space)) {
        //    active = false;
        //}

        if (debug)
            printf("Completed EBS acquisition.\n");
    }

    rateFile.close();
    davisHandle.dataStop();

    // Close automatically done by destructor.
    printf("Shutdown successful.\n");

    return (EXIT_SUCCESS);
}


void processing_threads(std::vector<StageController *>& ctrl, Buffers& buffers, const DBSCAN_KNN& T, const ProcessingInit& proc_init,
                        std::chrono::time_point<std::chrono::high_resolution_clock> start, const bool debug, const bool& active) {
    /*
    Primary processing thread.

    Args:
        ctrl:      Collection of controller pointers for each connected stage.
        T:         DBSCAN_KNN detector and tracker.
        proc_init: Globally important program parameters.
        start:     Program start time.
        debug:     Boolean debug flag.
        active:    Boolean indicating program status.

    Ret:
        None.

    Notes:
        Jacob - I've removed the tri-thread setup hoping that the new laptop has
        has the performance to keep up with the event packets.
    */

    std::ofstream detectionsFile(proc_init.event_file + "-detections.csv");
    std::ofstream eventFile(proc_init.event_file + "-events.csv");
    std::binary_semaphore update_positions(1);

    std::vector<int> panInit(ctrl.size(), 0);
    std::vector<int> tiltInit(ctrl.size(), 0);
    WindowInfo prev_trackingInfo;
    StageInfo prev_stageInfo(panInit, tiltInit);

    while (active) {
        if (debug)
            printf("Started event processing.");

        if (buffers.PacketQueue.empty())
            continue;

        std::future<WindowInfo> fut_resultA =
                std::async(std::launch::async, process_packet, buffers.PacketQueue.front(), T, proc_init,
                           prev_trackingInfo, std::ref(buffers.prev_positions), &update_positions, start);

        buffers.PacketQueue.pop();

        std::tie(prev_stageInfo, prev_trackingInfo) =
                read_future(ctrl, fut_resultA, proc_init, prev_stageInfo, detectionsFile, eventFile, start);

        if (debug)
            printf("Completed event processing.");
    }

    detectionsFile.close();
    eventFile.close();

    return;
}

cv::Mat formatYolov5(const cv::Mat& frame) {
    int row = frame.rows;
    int col = frame.cols;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    frame.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void camera_thread(StageCam& cam, StageController& ctrl, int height, int width, double hfovx, double hfovy,
                   const std::string& onnx_loc, bool enable_stage, bool enable_dnn, std::chrono::time_point<std::chrono::high_resolution_clock> start,
                   double confidence_thres, const bool debug, const bool &active, int idx) {
    /*
    Frame-based tracking.

    Args:
        cam:              Stage camera object. Provides images from frame-based sensors.
        ctrl:             Move the stages.
        height:           Height of FBS focal-plane arrays.
        width:            Width of FBS focal-plane arrays.
        hfovx:            Half FOV in x.
        hfovy:            Half FOV in y.
        onnx_loc:         Onnx file for Yolo.
        enable_stage:     Indiciates whether to enable stage.
        enable dnn:       Indicates whether to use a neural network.
        start:            Program start time.
        confdience_thres: Confidence level required to declare a detection.
        debug:            Just a debug flag.
        active:           Indicates whether the system should still be running.
        idx:              Indicates which camera to use.

    Ret:
        None.

    Notes:
        None.
    */

    std::vector<std::string> class_list{"drone"};
    cv::dnn::Net net;
    net = cv::dnn::readNet(onnx_loc);
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();

    while(active && cam.running()) {
        if (debug)
            printf("Started camera acquisition.");

        auto frame = cam.get_frame();
        cv::Mat color_frame;
        cv::cvtColor(frame, color_frame, cv::COLOR_GRAY2BGR);
        cv::Rect bbox;

        //if (key_is_pressed(XK_E)) {
        //    enable_dnn = !enable_dnn;
        //    if (enable_dnn)
        //        printf("Fine track enabled.\n");
        //    else
        //        printf("Fine track disabled.\n");
        //
        //    std::this_thread::sleep_for(std::chrono::milliseconds(100));
        //}

        if (enable_dnn) {
            if (!ctrl.get_tracker_status()) {
                cv::Mat input_image = formatYolov5(color_frame);  // making the image square
                cv::Mat blob = cv::dnn::blobFromImage(input_image, 1 / 255.0, cv::Size(640, 640), true);

                net.setInput(blob);
                cv::Mat predictions = net.forward();

                std::vector<int> class_ids;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes;

                cv::Mat output_data = predictions.reshape(0, predictions.size[1]);

                int image_width = input_image.cols;
                int image_height = input_image.rows;
                float x_factor = static_cast<float>(image_width) / 640;
                float y_factor = static_cast<float>(image_height) / 640;

                for (int r = 0; r < output_data.rows; r++) {
                    cv::Mat row = output_data.row(r);
                    float confidence = row.at<float>(4);
                    if (confidence >= confidence_thres) {
                        cv::Mat classes_scores = row.colRange(5, row.cols);
                        cv::Point max_loc;
                        cv::minMaxLoc(classes_scores, nullptr, nullptr, nullptr, &max_loc);
                        int class_id = max_loc.y;
                        if (classes_scores.at<float>(class_id) > 0.25) {
                            confidences.push_back(confidence);
                            class_ids.push_back(class_id);

                            float x = row.at<float>(0);
                            float y = row.at<float>(1);
                            float w = row.at<float>(2);
                            float h = row.at<float>(3);

                            int left = static_cast<int>((x - 0.5 * w) * x_factor);
                            int top = static_cast<int>((y - 0.5 * h) * y_factor);
                            int box_width = static_cast<int>(w * x_factor);
                            int box_height = static_cast<int>(h * y_factor);

                            cv::Rect box(left, top, box_width, box_height);
                            boxes.push_back(box);
                        }
                    }
                }

                std::vector<int> indexes;
                cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

                std::vector<int> result_class_ids;
                std::vector<float> result_confidences;
                std::vector<cv::Rect> result_boxes;

                if (indexes.size() >= 2) {
                    // Find the index of the box with the highest confidence
                    float max_confidence = confidences[indexes[0]];
                    int max_index = indexes[0];
                    for (int i = 1; i < indexes.size(); i++) {
                        int index = indexes[i];
                        float confidence = confidences[index];
                        if (confidence > max_confidence) {
                            max_confidence = confidence;
                            max_index = index;
                        }
                    }

                    // Swap the box with the highest confidence to the first position
                    std::swap(indexes[0], indexes[max_index]);
                }

                for (int i: indexes) {
                    result_confidences.push_back(confidences[i]);
                    result_class_ids.push_back(class_ids[i]);
                    result_boxes.push_back(boxes[i]);
                }

                for (size_t i = 0; i < result_class_ids.size(); i++) {
                    cv::Rect box = result_boxes[i];

                    cv::rectangle(color_frame, box, cv::Scalar(0, 255, 255), 2);
                    cv::rectangle(color_frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y),
                                  cv::Scalar(0, 255, 255), -1);
                    cv::putText(color_frame, "drone", cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 0, 0));
                }

                if (!result_boxes.empty()) {
                    bbox = result_boxes[0];
                    ctrl.activate_fine();
                    cv::Point2f originalCentroid((float)(bbox.x + bbox.width / 2.0), (float)(bbox.y + bbox.height / 2.0));
                    float scaleFactor = 1.2;
                    int newWidth = (int)((float)bbox.width * scaleFactor);
                    int newHeight = (int)((float)bbox.height * scaleFactor);

                    // Create a new rectangle with increased dimensions around the same centroid
                    cv::Rect newRect;
                    newRect.width = newWidth;
                    newRect.height = newHeight;
                    newRect.x = (int)(originalCentroid.x - newRect.width / 2.0);
                    newRect.y = (int)(originalCentroid.y - newRect.height / 2.0);
                    tracker->init(color_frame, newRect);
                }

            } else {
                bool isTrackingSuccessful = tracker->update(color_frame, bbox);
                if (isTrackingSuccessful) {// && !key_is_pressed(XK_Escape)) {
                    cv::rectangle(color_frame, bbox, cv::Scalar(255, 0, 0), 2);
                    double target_x = (double) bbox.x + (bbox.width / 2.0) - (width / 2.0);
                    double target_y = (height / 2.0) - (double) bbox.y - (bbox.height / 2.0);
                    int pan_inc = (int) (get_phi(target_x, width, hfovx) * 180.0 / M_PI / 0.02);
                    int tilt_inc = (int) (get_phi(target_y, height, hfovy) * 180.0 / M_PI / 0.02);
                    if (enable_stage)
                        ctrl.increment_setpoints(pan_inc, tilt_inc);
                } else {
                    ctrl.deactivate_fine();
                    tracker = cv::TrackerKCF::create();
                }
            }
        }

        // Calculate runtime performance.
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        int elapsed = (int)duration.count();

        // Save most recent image data.
        saveImage(color_frame, "./camera" + std::to_string(idx) + "_images", std::to_string(elapsed));
        cv::imshow("Camera" + std::to_string(idx), color_frame);

        if (debug)
            printf("Completed camera acquisition.");
    }
}
