#pragma once

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