# include <iostream>
# include <vector>
# include <string>
# include <sstream>

int main() {
     std::string stream {"15.333, -.234, 232.9, 0.33"};
     std::vector<double> res;

     std::stringstream s(stream);
     
     while (s.good()) {
          std::string sub;
          std::getline(s, sub, ',');
          res.push_back(std::stod(sub));
     }

     for (auto st : res)
          std::cout << st << "\n";

     return 0;
}
