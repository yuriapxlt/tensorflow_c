#include <iostream>
#include <numeric>
#include <iomanip>
#include <gtest/gtest.h>
#include <tensorflow/c/c_api.h>
#include "tfc.h"

namespace tf = ::tfc;

TEST(tfc, tensorflow)
{
    ASSERT_STREQ(TF_Version(), "1.15.2");
}

TEST(tfc, version)
{
    ASSERT_STREQ(tf::version(), "0.1");
}

TEST(tfc, gpu_fraction)
{
    ASSERT_THROW(tfc::gpu_fraction(0.), std::runtime_error);
    ASSERT_THROW(tfc::gpu_fraction(-0.1), std::runtime_error);
    ASSERT_THROW(tfc::gpu_fraction(1.), std::runtime_error);
    ASSERT_THROW(tfc::gpu_fraction(1.1), std::runtime_error);
    for (float f = 0.1; f < 1.; f += 0.1) ASSERT_NO_THROW(tfc::gpu_fraction(f));
}

TEST(tfc, session)
{
    tf::session pb("session.pb");
    pb.init();
    tf::tensor input_a{pb, "input_a"};
    tf::tensor input_b{pb, "input_b"};
    tf::tensor result{pb, "result"};
    std::size_t size = 10;
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.);
    input_a.set_data(data);
    input_b.set_data(data);
    ASSERT_EQ(input_a.get_data<float>().size(), input_b.get_data<float>().size());
    ASSERT_NO_THROW(pb.process({&input_a, &input_b}, result));
    ASSERT_EQ(result.get_data<float>().size(), size);
    int i(0);
    for (float f: result.get_data<float>()) ASSERT_FLOAT_EQ(f, 2 * i++);
}

TEST(tfc, shared_ptr)
{
    tf::session_ptr pb = std::make_shared<tf::session>("session.pb");
    pb->init();
    tf::tensor_ptr input_a = std::make_shared<tf::tensor>(*pb, "input_a");
    tf::tensor_ptr input_b = std::make_shared<tf::tensor>(*pb, "input_b");
    tf::tensor_ptr result = std::make_shared<tf::tensor>(*pb, "result");
    std::size_t size = 10;
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.);
    input_a->set_data(data);
    input_b->set_data(data);
    ASSERT_EQ(input_a->get_data<float>().size(), input_b->get_data<float>().size());
    ASSERT_NO_THROW(pb->process({input_a.get(), input_b.get()}, result.get()));
    ASSERT_EQ(result->get_data<float>().size(), size);
    int i(0);
    for (float f: result->get_data<float>()) ASSERT_FLOAT_EQ(f, 2 * i++);
}
