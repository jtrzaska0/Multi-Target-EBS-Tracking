# include <iostream>
# include <vector>
# include <armadillo>

int main() {
     size_t N;
     std::cin >> N;

     double * p {new double[N]};

     std::vector<double> ptr(20);
     for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 5; ++j)
               ptr[j + i * 5]  = j + i * 5;
     }

     arma::mat mat(ptr.data(), 4, 5);


     for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 5; ++j)
               std::cout << mat(i,j) << " ";
          std::cout << "\n";
     }

     return 0;
}
