#include "tfc.h"
#include <iostream>
#include <numeric>
#include <iomanip>

namespace tf = ::tfc;

int main(int argc, char* argv[]) 
{
    tf::Session session("session.pb");
    session.init();

    tf::Tensor input_a{session, "input_a"};
    tf::Tensor input_b{session, "input_b"};
    tf::Tensor result{session, "result"};

    std::vector<float> data(100);
    std::iota(data.begin(), data.end(), 0);

    input_a.set_data(data);
    input_b.set_data(data);

    session.run({&input_a, &input_b}, result);

    std::cout << std::endl;
    int i(0);
    for (float f : result.get_data<float>()) {
        std::cout << "\t" << f;
        if (!(++i % 10)) std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
