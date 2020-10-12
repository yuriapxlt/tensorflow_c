#include "tfc.h"
#include <iostream>
#include <numeric>
#include <iomanip>

namespace tf = ::tfc;

int main(int argc, char* argv[]) 
{
    tf::session pb("session.pb");
    std::cout << std::endl;
    pb.init();
    tf::tensor input_a{pb, "input_a"};
    tf::tensor input_b{pb, "input_b"};
    tf::tensor result{pb, "result"};
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 1);
    input_a.set_data(data);
    input_b.set_data(data);
    auto out = [](tf::tensor& t) { 
        std::ostringstream os; 
        for (auto d : t.get_data<float>()) os << d << " ";
        return os.str();
    };
    std::cout << "input_a [ " << out(input_a) << "]" << std::endl;
    std::cout << "input_b [ " << out(input_b) << "]" << std::endl;
    pb.process({&input_a, &input_b}, result);
    std::cout << "result [ " << out(result) << "]" << std::endl;
    return 0;
}
