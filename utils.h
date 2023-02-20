#pragma once

#include <iostream>
#include <cmath>

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

    int n_clusters = (int)centroids.n_cols;
    for (int i = 0; i < n_clusters; i++) {
        ret.push_back(centroids(0,i));
        ret.push_back(centroids(1,i));
    }

    // Process any position data that wasn't assigned.
    // First, create a new dataset with only the unassigned positions.
    // Then, loop through the list. If an unassigned point is within eps of another, add their median to the positions.
    // Otherwise, add the position individually.
    int n_unassigned = 0;
    for (int i = 0; i < (int)assignments.n_elem; i++) {
        if (assignments(i) == -1) {
            n_unassigned += 1;
        }
    }

    if (n_unassigned > 0) {
        arma::mat unassigned;
        unassigned.zeros(2, n_unassigned);
        int j = 0;
        for (int i = 0; i < (int) assignments.n_elem; i++) {
            if (assignments(i) == -1) {
                unassigned(0, j) = positions_mat(0, i);
                unassigned(1, j) = positions_mat(1, i);
                j += 1;
            }
        }

        if (n_unassigned >= 2) {
            mlpack::range::RangeSearch<> a(unassigned);
            std::vector<std::vector<size_t>> neighbors;
            std::vector<std::vector<double>> distances;
            mlpack::math::Range r(0.0, eps);
            a.Search(r, neighbors, distances);
            for (int i = 0; i < neighbors.size(); i++) {
                // Add the point to the list if it has no neighbors
                if (neighbors[i].empty()) {
                    ret.push_back(unassigned(0,i));
                    ret.push_back(unassigned(1,i));
                }
                // Add the median of the point and its neighbors to the list
                else {
                    std::vector<double> neighbor_x;
                    std::vector<double> neighbor_y;
                    for (auto ind:neighbors[i]) {
                        neighbor_x.push_back(unassigned(0,(int)ind));
                        neighbor_y.push_back(unassigned(1,(int)ind));
                    }
                    ret.push_back(median(neighbor_x, (int)neighbor_x.size()));
                    ret.push_back(median(neighbor_y, (int)neighbor_y.size()));
                }
            }
        }

        if (n_unassigned == 1) {
            ret.push_back(unassigned(0, 0));
            ret.push_back(unassigned(1, 0));
        }
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