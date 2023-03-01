#pragma once

#include <armadillo>
#include <mlpack.hpp>

void clear_arma(std::queue<arma::mat> &q) {
    std::queue<arma::mat> empty;
    std::swap(q, empty);
}

void clear_vector(std::queue<std::vector<double>> &q) {
    std::queue<std::vector<double>> empty;
    std::swap(q, empty);
}

void clear_cv(std::queue<cv::Mat> &q) {
    std::queue<cv::Mat> empty;
    std::swap(q, empty);
}

double update_average(int prev_val, int new_val, int n_samples) {
    return (prev_val*n_samples + new_val)/((double)n_samples + 1);
}

bool move_stage(float pan_position, float prev_pan_position, float tilt_position, float prev_tilt_position, double update) {
    float pan_change = abs((pan_position - prev_pan_position)/prev_pan_position);
    float tilt_change = abs((tilt_position - prev_tilt_position)/prev_tilt_position);
    if (pan_change > update || tilt_change > update)
        return true;
    return false;
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