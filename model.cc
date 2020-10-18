#include "tensorflow_c/api.h"
#include <numeric>
#include <iomanip>

namespace tf = ::tensorflow_c;

int main(int argc, char **argv)
{
    tf::session pb("model.pb", tf::gpu_fraction(0.1));
    tf::tensor num_detections_tensor{pb, "num_detections"};
    tf::tensor detection_scores_tensor{pb, "detection_scores"};
    tf::tensor detection_boxes_tensor{pb, "detection_boxes"};
    tf::tensor detection_classes_tensor{pb, "detection_classes"};
    tf::tensor image_tensor{pb, "image_tensor"};
    const size_t width = 10;
    const size_t height = 10;
    std::vector<uint8_t> data(1 * width * height * 3);
    std::iota(data.begin(), data.end(), 0.);
    image_tensor.set_data(data, {1, width, height, 3});
    pb.process(image_tensor, {&num_detections_tensor, &detection_scores_tensor, &detection_boxes_tensor, &detection_classes_tensor});
    int num_detections = (int)num_detections_tensor.get_data<float>()[0];
    std::cout << std::endl;
    for (int i = 0; i < num_detections; i++)
    {
        auto bbox_data = detection_boxes_tensor.get_data<float>();
        std::vector<float> bbox = {
            bbox_data[i * 4], 
            bbox_data[i * 4 + 1], 
            bbox_data[i * 4 + 2], 
            bbox_data[i * 4 + 3]
        };
        float detection_score = detection_scores_tensor.get_data<float>()[i];
        float location_xmin = bbox[1] * width;
        float location_ymin = bbox[0] * height;
        float location_width = (bbox[3] - bbox[1]) * width;
        float location_height = (bbox[2] - bbox[0]) * height;
        std::cout << "[" << i << "] "
            << "score[" << detection_score << "] "
            << "xmin[" << location_xmin << "] "
            << "ymin[" << location_ymin << "] "
            << "width[" << location_width << "] "
            << "height[" << location_height << "]" 
            << std::endl;
    }

    return 0;
}