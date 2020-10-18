#include <iostream>
#include <numeric>
#include <iomanip>
#include <gtest/gtest.h>
#include <tensorflow/c/c_api.h>
#include "tensorflow_c/api.h"

namespace tf = ::tensorflow_c;

TEST(tensorflow_c, version)
{
    ASSERT_STREQ(tf::version(), TF_Version());
}

TEST(tensorflow_c, gpu_fraction)
{
    ASSERT_THROW(tf::gpu_fraction(0.), std::runtime_error);
    ASSERT_THROW(tf::gpu_fraction(-0.1), std::runtime_error);
    ASSERT_THROW(tf::gpu_fraction(1.), std::runtime_error);
    ASSERT_THROW(tf::gpu_fraction(1.1), std::runtime_error);
    for (float f = 0.1; f < 1.; f += 0.1) ASSERT_NO_THROW(tf::gpu_fraction(f));
}

TEST(tensorflow_c, session)
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

TEST(tensorflow_c, shared_ptr)
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

TEST(tensorflow_c, model)
{
    tf::session pb("model.pb", tf::gpu_fraction(0.1));
    tf::tensor num_detections_tensor{pb, "num_detections"};
    tf::tensor detection_scores_tensor{pb, "detection_scores"};
    tf::tensor detection_boxes_tensor{pb, "detection_boxes"};
    tf::tensor detection_classes_tensor{pb, "detection_classes"};
    tf::tensor image_tensor{pb, "image_tensor"};
    const size_t size = 10;
    std::vector<uint8_t> data(1 * size * size * 3);
    std::iota(data.begin(), data.end(), 0.);
    image_tensor.set_data(data, {1, size, size, 3});
    pb.process(image_tensor, {&num_detections_tensor, &detection_scores_tensor, &detection_boxes_tensor, &detection_classes_tensor});
    int num_detections = (int)num_detections_tensor.get_data<float>()[0];
    ASSERT_EQ(num_detections, 3);
    for (int i = 0; i < num_detections; i++)
    {
        auto bbox_data = detection_boxes_tensor.get_data<float>();
        std::vector<float> bbox = {
            bbox_data[i * 4 + 0], 
            bbox_data[i * 4 + 1], 
            bbox_data[i * 4 + 2], 
            bbox_data[i * 4 + 3]
        };
        float detection_score = detection_scores_tensor.get_data<float>()[i];
        ASSERT_FLOAT_EQ(detection_score, 0.1 * (i + 1));
        float location_xmin = bbox[1] * size;
        ASSERT_FLOAT_EQ(location_xmin, 2 * i + 1);
        float location_ymin = bbox[0] * size;
        ASSERT_FLOAT_EQ(location_ymin, 2 * i + 1);
        float location_width = (bbox[3] - bbox[1]) * size;
        ASSERT_FLOAT_EQ(location_width, 1);
        float location_height = (bbox[2] - bbox[0]) * size;
        ASSERT_FLOAT_EQ(location_height, 1);
    }
}
