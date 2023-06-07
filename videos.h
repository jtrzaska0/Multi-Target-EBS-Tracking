#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>

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

    // Create a VideoWriter object to write the video
    cv::Size frameSize;
    cv::Mat frame = cv::imread(imagePaths[0], cv::IMREAD_COLOR);
    if (!frame.empty()) {
        frameSize = frame.size();
    } else {
        std::cerr << "Failed to read the first image." << std::endl;
        return;
    }
    auto first_time = (double)extractFileNameAsInt(imagePaths[0]);
    auto last_time = (double)extractFileNameAsInt(imagePaths[imagePaths.size() - 1]);
    if (first_time < 0 || last_time < 0)
        return;

    double ms_per_frame = 1000.0 / fps;
    double elapsed_time = first_time;
    cv::VideoWriter videoWriter(outputVideoPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frameSize);

    while (elapsed_time <= last_time) {
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
        elapsed_time += ms_per_frame;
    }

    videoWriter.release();
    std::cout << "Video created: " << outputVideoPath << std::endl;
}
