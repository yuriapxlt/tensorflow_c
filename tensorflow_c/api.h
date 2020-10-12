#ifndef __TENSORFLOW_C_API_H__
#define __TENSORFLOW_C_API_H__

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <tensorflow/c/c_api.h>

namespace tensorflow_c
{

const char* version();
std::vector<uint8_t> gpu_fraction(double fraction);

class session;

class tensor 
{
public:
    friend class session;
    tensor(const session& pb, const std::string& operation);
    tensor(const tensor& tensor) = delete;
    tensor(tensor&& tensor) = default;
    tensor& operator=(const tensor& tensor) = delete;
    tensor& operator=(tensor&& tensor) = default;
    ~tensor();
    void clean();
    template<typename T> void set_data(std::vector<T> new_data);
    template<typename T> void set_data(std::vector<T> new_data, const std::vector<int64_t>& new_shape);
    template<typename T> std::vector<T> get_data();
	std::vector<int64_t> get_shape();
private:
    TF_Tensor* val;
    TF_Output op;
    TF_DataType type;
    std::vector<int64_t> shape;
    std::unique_ptr<std::vector<int64_t>> actual_shape;
    void* data;
    int flag;
    void error_check(bool condition, const std::string& error);
    template <typename T> static TF_DataType deduce_type();
    void deduce_shape();
};
typedef std::shared_ptr<tensor> tensor_ptr;

class session 
{
public:
    friend class tensor;
    explicit session(const std::string& pb_filename, const std::vector<uint8_t>& config_options = {});
    session(const session& pb) = delete;
    session(session&& pb) = default;
    session& operator=(const session& pb) = delete;
    session& operator=(session&& pb) = default;
    ~session();
    void init();
    std::vector<std::string> get_operations() const;
    void process(const std::vector<tensor*>& inputs, const std::vector<tensor*>& outputs);
    void process(tensor& input, const std::vector<tensor*>& outputs);
    void process(const std::vector<tensor*>& inputs, tensor& output);
    void process(tensor& input, tensor& output);
    void process(tensor* input, const std::vector<tensor*>& outputs);
    void process(const std::vector<tensor*>& inputs, tensor* output);
    void process(tensor* input, tensor* output);
private:
    TF_Graph* graph;
    TF_Session* pb;
    TF_Status* status;
    static TF_Buffer* read(const std::string& filename);
    bool status_check(bool throw_exc) const;
    void error_check(bool condition, const std::string &error) const;
};
typedef std::shared_ptr<session> session_ptr;

}

#endif
