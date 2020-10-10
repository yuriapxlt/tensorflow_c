#include <iostream>
#include <numeric>
#include <iomanip>
#include <gtest/gtest.h>
#include <tensorflow/c/c_api.h>
#include "tfc.h"

namespace tf = ::tfc;

TEST(tfc, TensorFlow)
{
    ASSERT_STREQ(TF_Version(), "1.15.2");
}

TEST(tfc, version)
{
    ASSERT_STREQ(tf::version(), "0.1");
}

TEST(tfc, session)
{
    tf::Session session("session.pb");
    session.init();

    tf::Tensor input_a{session, "input_a"};
    tf::Tensor input_b{session, "input_b"};
    tf::Tensor result{session, "result"};

    std::size_t size = 100;

    std::vector<float> a_data(size);
    std::iota(a_data.begin(), a_data.end(), 0.1);
    input_a.set_data(a_data);

    std::vector<float> b_data(size);
    std::iota(b_data.begin(), b_data.end(), 0.2);
    input_b.set_data(b_data);

    ASSERT_EQ(input_a.get_data<float>().size(), input_b.get_data<float>().size());

    ASSERT_NO_THROW(session.run({&input_a, &input_b}, result));

    ASSERT_EQ(result.get_data<float>().size(), size);

    int i(0);
    for (float f: result.get_data<float>()) 
    {
        ASSERT_FLOAT_EQ(f, 0.3 + 2 * i++);
    }
}
