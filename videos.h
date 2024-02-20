#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>
#include "progressbar.h"

void saveImage(const cv::Mat& frame, const std::string& folderPath, const std::string& fileName) {
    std::string filePath = folderPath + "/" + fileName + ".jpg";
    cv::imwrite(filePath, frame);
}

int extractFileNameAsInt(const std::string& filePath) {
    std::string fileName = std::filesystem::path(filePath).stem().string();
    try {
        return std::stoi(fileName);
    } catch (const std::exception& e) {
        std::cerr << "Failed to convert file name to integer: " << e.what() << std::endl;
        return -1;
    }
}

std::pair<int, int> getMinMaxTimes(const std::vector<std::string>& file_paths) {
    int min_file_name = std::numeric_limits<int>::max();
    int max_file_name = std::numeric_limits<int>::min();

    for (const std::string& file_path : file_paths) {
        int file_name_as_int = extractFileNameAsInt(file_path);
        min_file_name = std::min(min_file_name, file_name_as_int);
        max_file_name = std::max(max_file_name, file_name_as_int);
    }

    if (min_file_name == std::numeric_limits<int>::max() || max_file_name == std::numeric_limits<int>::min()) {
        return std::make_pair(-1, -1);
    }

    return std::make_pair(min_file_name, max_file_name);
}

void createVideoFromImages(const std::string& directoryPath, const std::string& outputVideoPath, double fps) {
    std::vector<std::string> imagePaths;

    // Read all image paths from the directory
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            imagePaths.push_back(entry.path().string());
        }
    }

    // Sort the image paths based on filenames
    std::sort(imagePaths.begin(), imagePaths.end());

    if (imagePaths.empty()) {
        std::cerr << "Failed to create " << outputVideoPath << ". No images found.\n";
        return;
    }

    // Create a VideoWriter object to write the video
    cv::Size frameSize;
    cv::Mat frame = cv::imread(imagePaths[0], cv::IMREAD_COLOR);
    if (!frame.empty()) {
        frameSize = frame.size();
    } else {
        std::cerr << "Failed to read the first image." << std::endl;
        return;
    }
    const auto [first_time, last_time] = getMinMaxTimes(imagePaths);
    if (first_time < 0 || last_time < 0)
        return;

    double ms_per_frame = 1000.0 / fps;
    double elapsed_time = first_time;
    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frameSize);
    progressbar bar((int)((last_time - first_time) / ms_per_frame) + 1);

    while (elapsed_time <= (double)last_time) {
        // Find the image with the filename closest to elapsed_time
        std::string closestImagePath;
        double closestDiff = std::numeric_limits<double>::max();

        for (const auto& imagePath : imagePaths) {
            int fileNameAsInt = extractFileNameAsInt(imagePath);
            if (fileNameAsInt >= 0) {
                double diff = std::abs((double)fileNameAsInt - elapsed_time);
                if (diff < closestDiff) {
                    closestDiff = diff;
                    closestImagePath = imagePath;
                }
            }
        }

        // Append the closest image if found
        if (!closestImagePath.empty()) {
            frame = cv::imread(closestImagePath, cv::IMREAD_COLOR);
            if (!frame.empty()) {
                videoWriter.write(frame);
            } else {
                std::cerr << "Failed to read image: " << closestImagePath << std::endl;
            }
        }
        bar.update();
        elapsed_time += ms_per_frame;
    }

    std::cerr << "\n";
    videoWriter.release();
    std::cerr << "Video created: " << outputVideoPath << std::endl;
}
