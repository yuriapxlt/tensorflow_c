#ifndef __TFC_H__
#define __TFC_H__

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <tensorflow/c/c_api.h>

namespace tfc
{

const char* version();
std::vector<uint8_t> gpu_fraction(double fraction);

class Tensor;
class Session;

class Tensor {
public:
    friend class Session;
    Tensor(const Session& session, const std::string& operation);
    Tensor(const Tensor &tensor) = delete;
    Tensor(Tensor &&tensor) = default;
    Tensor& operator=(const Tensor &tensor) = delete;
    Tensor& operator=(Tensor &&tensor) = default;
    ~Tensor();
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

class Session {
public:
    friend class Tensor;
    explicit Session(const std::string& session_filename, const std::vector<uint8_t>& config_options = {});
    Session(const Session &session) = delete;
    Session(Session &&session) = default;
    Session& operator=(const Session &session) = delete;
    Session& operator=(Session &&session) = default;
    ~Session();
    void init();
    void restore(const std::string& ckpt);
    void save(const std::string& ckpt);
    std::vector<std::string> get_operations() const;
    void run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    void run(Tensor& input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor& output);
    void run(Tensor& input, Tensor& output);
    void run(Tensor* input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor* output);
    void run(Tensor* input, Tensor* output);
private:
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;
    static TF_Buffer* read(const std::string& filename);
    bool status_check(bool throw_exc) const;
    void error_check(bool condition, const std::string &error) const;
};

}

#endif
