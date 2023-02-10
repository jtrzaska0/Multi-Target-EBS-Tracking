# include <iostream>
# include <vector>
# include <string>
# include "../eigen-3.4.0/Eigen/Dense"
# include <cmath>
# include <set>
# include <algorithm>
# include <iterator>
# include <map>


void f(double x) {

     std::cout << x << "\n";

     return;
}

void eigentest() {

     Eigen::MatrixXd x(3,1);
     x(0,0) = 3;
     x(1,0) = 0;
     x(2,0) = 0;
     Eigen::MatrixXd m {Eigen::MatrixXd::Ones(3,3)};
     Eigen::MatrixXd y = x.reshaped(1, 3);
     std::cout << x << "\n";
    
     double s = (x.transpose() * m * x)(0,0);

     std::cout << s << std::endl;

     Eigen::MatrixXd b = Eigen::MatrixXd::Ones(1,5);
     std::cout << b.cols() << std::endl;
     std::cout << b.rows() << std::endl;

     return;
}


void settest() {

     std::set<int> a {15, 45, 1, 4};
     std::set<int>::iterator itr;
     std::set<int> c0 {1, 5, 3, 22, 332, 4};
     std::set<int> c1 {1, 5, 13, 12, 43, 332};
     std::set<int> c;
     std::set_difference(c0.begin(), c0.end(), c1.begin(), c1.end(), std::inserter(c, c.end()));

     for (auto& itr : c)
          std::cout << itr << "\n";
}

void maptest() {
     std::map<int, Eigen::MatrixXd> map;

     map.insert(std::make_pair(0, Eigen::MatrixXd::Ones(2,2)));
     map.insert(std::make_pair(1, Eigen::MatrixXd::Zero(1,1)));
     map.insert(std::make_pair(2, Eigen::MatrixXd::Random(3,3)));
     
     // Loop over the object keys.
     for (int i {0}; i < map.size(); ++i)
          std::cout << map[i] << std::endl;;

     std::cout << "\n\n";

     // Loop over the object values
     for (std::map<int, Eigen::MatrixXd>::iterator itr = map.begin(); itr != map.end(); ++itr)
          std::cout << itr->second << std::endl;


     std::cout << "---------------------------------------------\n\n";

     std::map<int, int> nums;

     nums.insert(std::make_pair(0, 15));
     std::cout << nums[0] << "\n";
     for (std::map<int, int>::iterator it = nums.begin(); it != nums.end(); ++it) {
          nums[it->first]++;
          std::cout << nums[it->first] << "\n";
     }

     std::cout << nums.size() << "\n";
}


void eigen_w_stdset() {

     std::vector<int> ridx {0, 4, 3};
     Eigen::MatrixXd mat = Eigen::MatrixXd::Random(5,5);

     std::cout << mat << "\n\n";
     std::cout << mat(ridx, Eigen::all) <<"\n\n";
}

void makeRange() {
     Eigen::VectorXi x = Eigen::VectorXi::LinSpaced(10, 0, 9);
     std::cout << x << std::endl;
}

void matInit() {
     Eigen::MatrixXd mat {{-2, 4,}, {1, 3}};
     std::cout << mat << "\n\n";

}

void idxTest() {
     std::map<int, int> a;

     a[1] = 15;
     for (std::map<int, int>::iterator iter = a.begin(); iter != a.end(); iter++)
          std::cout << iter->first << " " << iter->second << "\n";
}

void slicingTest() {
     Eigen::MatrixXd mat {Eigen::MatrixXd::Random(5,5)};
     Eigen::MatrixXd A   {{-1},{3},{5},{2},{0}};

     std::cout << mat << "\n\n";

     mat(2, Eigen::all) = A.reshaped(1, 5);
     std::cout << mat << "\n\n";
}

void getRow() {

     Eigen::MatrixXd X {Eigen::MatrixXd::Random(5,5)};
     std::cout << X << "\n\n";
     std::cout << X.row(1) << "\n\n";
     std::cout << X.row(1).minCoeff() << "\n\n";
}

void sortMap() {
     std::map<int, int> map;

     Eigen::MatrixXi K {Eigen::MatrixXi::Random(1, 15)};
     Eigen::MatrixXi V {Eigen::MatrixXi::Random(1, 15)};
     for (int i {0}; i < 15; ++i) {
          map.insert(std::make_pair(K(0,i), i));
     }

     for (std::map<int, int>::iterator iter = map.begin(); iter != map.end(); ++iter)
          std::cout << iter->first << " " << iter->second << "\n\n";

}

void useTest() {
     using df = std::map<std::string, int>;

     df tester {std::make_pair("test", 5), std::make_pair("one", 11)};

     std::cout << tester["test"] << " " << tester["one"] << "\n";
     std::cout << tester.size() << "\n";

//     for (std::map<std::string, int>::iterator iter = tester.begin(); iter != tester.end(); ++iter)
  //        std::cout << iter->first << " " << iter->second << "\n\n";
}


void pairTest() {

     std::pair<std::string, std::string> pair {std::make_pair("test", "test1")};
     std::cout << pair.first;
     std::cout << " " << pair.second << "\n";

     pair.first = "test2";
     std::cout << pair.first;
     std::cout << " " << pair.second << "\n";
}


struct TESTSTRUCT {
     std::string name;
     std::pair<int, int> p;
};

void structTest() {

     TESTSTRUCT ts {.name="structo", .p=std::make_pair(-2, 4)};
     std::cout << ts.name << "\n";
     std::cout << ts.p.first << " " << ts.p.second << "\n";
}

void equalTest() {
     Eigen::MatrixXd a {Eigen::MatrixXd::Constant(11.5, 2,3)};
     Eigen::MatrixXd b {a};
     b(0,0) = -1543;

     std::cout << a << "\n" << b << "\n";

     return;
}

void dotTest() {
     Eigen::Vector4f x {1, 0, 2, 3};
     std::cout << x.dot(x) << "\n";
     return;
}

void emptyMatrixTest() {

     Eigen::MatrixXd a;

     std::cout << "a.rows() = " << a.rows() << "\n"
               << "a.cols() = " << a.cols()
               << std::endl;

     return;
}

int main(int arg, char ** argv) {
     //eigentest();

     //f(15);

     //settest();

     //maptest();

     //eigen_w_stdset();

     //makeRange();

     //matInit();

     //idxTest();

     //slicingTest();

    // getRow();

//     sortMap();

//     useTest();

//     pairTest();

//     structTest();

//     equalTest();

//     dotTest();

     emptyMatrixTest();

     return 0;
}
