#pragma once

#include <tuple>

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

std::tuple<double, double> get_position(const std::string& method, std::vector<double> positions) {
    std::tuple<double, double> ret = {0, 0};
    std::vector<double> xs;
    std::vector<double> ys;
    bool toggle = false;
    std::partition_copy(positions.begin(),
                        positions.end(),
                        std::back_inserter(xs),
                        std::back_inserter(ys),
                        [&toggle](int) { return toggle = !toggle; });
    int size = (int)xs.size();

    if (method == "median") {
        ret = {median(xs, size), median(ys, size)};
    }
    return ret;
}