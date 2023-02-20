#pragma once

#include <armadillo>
#include <mlpack.hpp>
#include <utility>

double median(std::vector<double> a, int n) {
    // Even number of elements
    if (n % 2 == 0) {
        nth_element(a.begin(),a.begin() + n / 2,a.end());
        nth_element(a.begin(),a.begin() + (n - 1) / 2,a.end());
        return (a[(n - 1) / 2] + a[n / 2]) / 2;
    }
    else {
        nth_element(a.begin(),a.begin() + n / 2,a.end());
        return a[n / 2];
    }
}

arma::mat get_matrix(std::vector<double> xs, std::vector<double> ys) {
    arma::mat ret;
    ret.zeros(2, xs.size());
    for (int i = 0; i < xs.size(); i++) {
        ret(0, i) = xs[i];
        ret(1, i) = ys[i];
    }
    return ret;
}

std::vector<double> get_kmeans_positions(const std::vector<double>& xs, std::vector<double> ys) {
    std::vector<double> ret;
    mlpack::KMeans<> k;
    arma::mat positions_mat = get_matrix(xs, std::move(ys));
    arma::Row<size_t> assignments;
    arma::mat centroids;
    int nclusters = std::min((int)xs.size(), 3);
    k.Cluster(positions_mat, nclusters, assignments, centroids);

    for (int i = 0; i < nclusters; i++) {
        ret.push_back(centroids(0,i));
        ret.push_back(centroids(1,i));
    }
    return ret;
}

std::vector<double> get_dbscan_positions(const std::vector<double>& xs, std::vector<double> ys, double eps) {
    std::vector<double> ret;
    mlpack::dbscan::DBSCAN<> db(eps, 3);
    arma::mat positions_mat = get_matrix(xs, std::move(ys));
    arma::Row<size_t> assignments;
    arma::mat centroids;
    db.Cluster(positions_mat, assignments, centroids);

    int nclusters = (int)centroids.n_cols;
    for (int i = 0; i < nclusters; i++) {
        ret.push_back(centroids(0,i));
        ret.push_back(centroids(1,i));
    }
    return ret;
}

std::vector<double> get_position(const std::string& method, std::vector<double> positions, double eps) {
    std::vector<double> ret;
    if (!positions.empty()) {
        std::vector<double> xs;
        std::vector<double> ys;
        bool toggle = false;
        std::partition_copy(positions.begin(),
                            positions.end(),
                            std::back_inserter(xs),
                            std::back_inserter(ys),
                            [&toggle](int) { return toggle = !toggle; });
        int size = (int) xs.size();

        if (method == "median") {
            ret.push_back(median(xs, size));
            ret.push_back(median(ys, size));
            return ret;
        }
        if (method == "dbscan") {
            ret = get_dbscan_positions(xs, ys, eps);
            return ret;
        }
        if (method == "kmeans") {
            ret = get_kmeans_positions(xs, ys);
            return ret;
        }
    }
    return ret;
}