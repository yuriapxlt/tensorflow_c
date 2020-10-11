#include "tfc.h"
#include <fstream>
#include <utility>

namespace tfc
{

const char* version() 
{
    return "0.1";
}

void error_check(bool condition, const std::string &error) 
{
    if (!condition) throw std::runtime_error(error);
}

std::vector<uint8_t> gpu_fraction(double fraction)
{
    error_check((fraction > 0. && fraction < 1.), "invalid gpu fraction");
    std::vector<uint8_t> options{ 0x32, 0x09, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0, 0x0 };
    uint8_t* bytes = (uint8_t*)(&fraction);
    for (int i = 0; i < 8; ++i) options[3 + i] = bytes[i];
    return options;
}

Tensor::Tensor(const Session& session, const std::string& operation) 
{
    this->op.oper = TF_GraphOperationByName(session.graph, operation.c_str());
    this->op.index = 0;
    error_check(this->op.oper != nullptr, "invalid operation [" + operation + "]" );
    int n_dims = TF_GraphGetTensorNumDims(session.graph, this->op, session.status);
    this->type = TF_OperationOutputType(this->op);
    if (n_dims > 0) 
    {
        auto *dims = new int64_t[n_dims];
        TF_GraphGetTensorShape(session.graph, this->op, dims, n_dims, session.status);
        session.status_check(true);
        this->shape = std::vector<int64_t>(dims, dims + n_dims);
        delete[] dims;
    }
    this->flag = 0;
    this->val = nullptr;
    this->data = nullptr;
}

Tensor::~Tensor() 
{
    this->clean();
}

void Tensor::clean() 
{
    if (this->flag == 1) 
    {
        TF_DeleteTensor(this->val);
        this->flag = 0;
    }
    this->data = nullptr;
}

void  Tensor::error_check(bool condition, const std::string &error) 
{
    if (condition) return;
    this->flag = -1;
    throw std::runtime_error(error);
}

template<typename T> void Tensor::set_data(std::vector<T> new_data) 
{
    if (this->flag == 1) 
    {
        TF_DeleteTensor(this->val);
        this->flag = 0;
    }
    this->error_check(this->flag != -1, "invalid tensor");
    this->error_check(deduce_type<T>() == this->type, "invalid tensor type");
    this->error_check(!this->shape.empty(), "invalid tensor shape");
    this->error_check(std::count(this->shape.begin(), this->shape.end(), -1) >= -1, "invalid shape dimension");
    auto exp_size = std::abs(std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<int64_t>()));
    this->error_check(new_data.size() % exp_size == 0, "invalide elements number");
    auto d = [](void* ddata, size_t, void*) {free(static_cast<T*>(ddata));};
    this->actual_shape = std::make_unique<decltype(actual_shape)::element_type>(shape.begin(), shape.end());
    std::replace_if (actual_shape->begin(), actual_shape->end(), [](int64_t r) {return r==-1;}, new_data.size()/exp_size);
    this->data = malloc(sizeof(T) * new_data.size());
    memcpy(this->data, new_data.data(), sizeof(T) * new_data.size());
    this->val = TF_NewTensor(this->type, actual_shape->data(), actual_shape->size(), this->data, sizeof(T) * new_data.size(), d, nullptr);
    this->error_check(this->val != nullptr, "tensor allocation failed");
    this->flag = 1;
}

template<typename T> void Tensor::set_data(std::vector<T> new_data, const std::vector<int64_t>& new_shape) 
{
    this->error_check(this->shape.empty() || this->shape.size() == new_shape.size(), "shape mismatch");
    auto old_shape = this->shape;
    this->shape = new_shape;
    this->set_data(new_data);
    this->shape = old_shape;
}

template<typename T> std::vector<T> Tensor::get_data() 
{
    this->error_check(this->flag != -1, "invalid tensor");
    this->error_check(deduce_type<T>() == this->type, "invalid tensor type");
    this->error_check(this->flag != 0, "empty tensor");
    auto raw_data = TF_TensorData(this->val);
    this->error_check(raw_data != nullptr, "empty tensor data");
    size_t size = TF_TensorByteSize(this->val) / TF_DataTypeSize(TF_TensorType(this->val));
    const auto T_data = static_cast<T*>(raw_data);
    return std::vector<T>(T_data, T_data + size);
}

std::vector<int64_t> Tensor::get_shape() 
{
	return shape;
}

template<typename T> TF_DataType Tensor::deduce_type() 
{
    if (std::is_same<T, float>::value)
        return TF_FLOAT;
    if (std::is_same<T, double>::value)
        return TF_DOUBLE;
    if (std::is_same<T, int32_t >::value)
        return TF_INT32;
    if (std::is_same<T, uint8_t>::value)
        return TF_UINT8;
    if (std::is_same<T, int16_t>::value)
        return TF_INT16;
    if (std::is_same<T, int8_t>::value)
        return TF_INT8;
    if (std::is_same<T, int64_t>::value)
        return TF_INT64;
    if (std::is_same<T, uint16_t>::value)
        return TF_UINT16;
    if (std::is_same<T, uint32_t>::value)
        return TF_UINT32;
    if (std::is_same<T, uint64_t>::value)
        return TF_UINT64;
    throw std::runtime_error{"invalid tensor type"};
}

void Tensor::deduce_shape() 
{
    int n_dims = TF_NumDims(this->val);
    if (n_dims > 0) 
    {
        this->shape = std::vector<int64_t>(n_dims, -1);
        for (int i=0; i<n_dims; i++) this->shape[i] = TF_Dim(this->val, i);
    }
}

template TF_DataType Tensor::deduce_type<float>();
template TF_DataType Tensor::deduce_type<double>();
template TF_DataType Tensor::deduce_type<int8_t>();
template TF_DataType Tensor::deduce_type<int16_t>();
template TF_DataType Tensor::deduce_type<int32_t>();
template TF_DataType Tensor::deduce_type<int64_t>();
template TF_DataType Tensor::deduce_type<uint8_t>();
template TF_DataType Tensor::deduce_type<uint16_t>();
template TF_DataType Tensor::deduce_type<uint32_t>();
template TF_DataType Tensor::deduce_type<uint64_t>();

template std::vector<float> Tensor::get_data<float>();
template std::vector<double> Tensor::get_data<double>();
template std::vector<bool> Tensor::get_data<bool>();
template std::vector<int8_t> Tensor::get_data<int8_t>();
template std::vector<int16_t> Tensor::get_data<int16_t>();
template std::vector<int32_t> Tensor::get_data<int32_t>();
template std::vector<int64_t> Tensor::get_data<int64_t>();
template std::vector<uint8_t> Tensor::get_data<uint8_t>();
template std::vector<uint16_t> Tensor::get_data<uint16_t>();
template std::vector<uint32_t> Tensor::get_data<uint32_t>();
template std::vector<uint64_t> Tensor::get_data<uint64_t>();

template void Tensor::set_data<float>(std::vector<float> new_data);
template void Tensor::set_data<double>(std::vector<double> new_data);
template void Tensor::set_data<int8_t>(std::vector<int8_t> new_data);
template void Tensor::set_data<int16_t>(std::vector<int16_t> new_data);
template void Tensor::set_data<int32_t>(std::vector<int32_t> new_data);
template void Tensor::set_data<int64_t>(std::vector<int64_t> new_data);
template void Tensor::set_data<uint8_t>(std::vector<uint8_t> new_data);
template void Tensor::set_data<uint16_t>(std::vector<uint16_t> new_data);
template void Tensor::set_data<uint32_t>(std::vector<uint32_t> new_data);
template void Tensor::set_data<uint64_t>(std::vector<uint64_t> new_data);

template void Tensor::set_data<float>(std::vector<float> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<double>(std::vector<double> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int8_t>(std::vector<int8_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int16_t>(std::vector<int16_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int32_t>(std::vector<int32_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<int64_t>(std::vector<int64_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint8_t>(std::vector<uint8_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint16_t>(std::vector<uint16_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint32_t>(std::vector<uint32_t> new_data, const std::vector<int64_t>& new_shape);
template void Tensor::set_data<uint64_t>(std::vector<uint64_t> new_data, const std::vector<int64_t>& new_shape);

Session::Session(const std::string& session_filename, const std::vector<uint8_t>& config_options) 
{
    this->status = TF_NewStatus();
    this->graph = TF_NewGraph();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    if (!config_options.empty())
    {
        TF_SetConfig(sess_opts, static_cast<const void*>(config_options.data()), config_options.size(), this->status);
        this->status_check(true);
    }
    this->session = TF_NewSession(this->graph, sess_opts, this->status);
    TF_DeleteSessionOptions(sess_opts);
    this->status_check(true);
    TF_Graph* g = this->graph;
    TF_Buffer* def = read(session_filename);
    this->error_check(def != nullptr, "session read error");
    TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(g, def, graph_opts, this->status);
    TF_DeleteImportGraphDefOptions(graph_opts);
    TF_DeleteBuffer(def);
    this->status_check(true);
}

Session::~Session() 
{
    TF_DeleteSession(this->session, this->status);
    TF_DeleteGraph(this->graph);
    this->status_check(true);
    TF_DeleteStatus(this->status);
}

void Session::init() 
{
    TF_Operation* init_op[1] = {TF_GraphOperationByName(this->graph, "init")};
    this->error_check(init_op[0]!= nullptr, "no operation named \"init\" exists");
    TF_SessionRun(this->session, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0, init_op, 1, nullptr, this->status);
    this->status_check(true);
}

void Session::save(const std::string &ckpt) 
{
#ifdef TENSORFLOW_C_TF_TSTRING_H_
    std::unique_ptr<TF_TString, decltype(&TF_TString_Dealloc)> tstr(new TF_TString, &TF_TString_Dealloc);
    TF_TString_Copy(tstr.get(), ckpt.c_str(), ckpt.size());
    auto deallocator = [](void* data, size_t len, void* arg) {};
    TF_Tensor* t = TF_NewTensor(TF_STRING, nullptr, 0, tstr.get(), 1, deallocator, nullptr);
#else
    size_t size = 8 + TF_StringEncodedSize(ckpt.length());
    TF_Tensor* t = TF_AllocateTensor(TF_STRING, nullptr, 0, size);
    char* data = static_cast<char *>(TF_TensorData(t));
    for (int i=0; i<8; i++) {data[i]=0;}
    TF_StringEncode(ckpt.c_str(), ckpt.size(), data + 8, size - 8, status);
    memset(data, 0, 8);
    TF_StringEncode(ckpt.c_str(), ckpt.length(), (char*)(data + 8), size - 8, status);
#endif 
    if (!this->status_check(false)) 
    {
        TF_DeleteTensor(t);
        std::cerr << "error during filename " << ckpt << " encoding" << std::endl;
        this->status_check(true);
    }
    TF_Output output_file;
    output_file.oper = TF_GraphOperationByName(this->graph, "save/Const");
    output_file.index = 0;
    TF_Output inputs[1] = {output_file};
    TF_Tensor* input_values[1] = {t};
    const TF_Operation* restore_op[1] = {TF_GraphOperationByName(this->graph, "save/control_dependency")};
    if (!restore_op[0]) {
        TF_DeleteTensor(t);
        this->error_check(false, "no operation named \"save/control_dependencyl\" exists");
    }
    TF_SessionRun(this->session, nullptr, inputs, input_values, 1, nullptr, nullptr, 0, restore_op, 1, nullptr, this->status);
    TF_DeleteTensor(t);
    this->status_check(true);
}

void Session::restore(const std::string& ckpt) {
#ifdef TENSORFLOW_C_TF_TSTRING_H_
    std::unique_ptr<TF_TString, decltype(&TF_TString_Dealloc)> tstr(new TF_TString, &TF_TString_Dealloc);
    TF_TString_Copy(tstr.get(), ckpt.c_str(), ckpt.size());
    auto deallocator = [](void* data, size_t len, void* arg) {};
    TF_Tensor* t = TF_NewTensor(TF_STRING, nullptr, 0, tstr.get(), 1, deallocator, nullptr);
#else
    size_t size = 8 + TF_StringEncodedSize(ckpt.size());
    TF_Tensor* t = TF_AllocateTensor(TF_STRING, nullptr, 0, size);
    char* data = static_cast<char *>(TF_TensorData(t));
    for (int i=0; i<8; i++) {data[i]=0;}
    TF_StringEncode(ckpt.c_str(), ckpt.size(), data + 8, size - 8, status);
#endif
    if (!this->status_check(false)) 
    {
        TF_DeleteTensor(t);
        std::cerr << "error during filename " << ckpt << " encoding" << std::endl;
        this->status_check(true);
    }
    TF_Output output_file;
    output_file.oper = TF_GraphOperationByName(this->graph, "save/Const");
    output_file.index = 0;
    TF_Output inputs[1] = {output_file};
    TF_Tensor* input_values[1] = {t};
    const TF_Operation* restore_op[1] = {TF_GraphOperationByName(this->graph, "save/restore_all")};
    if (!restore_op[0]) 
    {
        TF_DeleteTensor(t);
        this->error_check(false, "error: No operation named \"save/restore_all\" exists");
    }
    TF_SessionRun(this->session, nullptr, inputs, input_values, 1, nullptr, nullptr, 0, restore_op, 1, nullptr, this->status);
    TF_DeleteTensor(t);
    this->status_check(true);
}

TF_Buffer* Session::read(const std::string& filename) 
{
    std::ifstream file (filename.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) 
    {
        std::cerr << "error open [" << filename << "]" << std::endl;
        return nullptr;
    }
    auto size = file.tellg();
    file.seekg (0, std::ios::beg);
    auto data = std::make_unique<char[]>(size);
    file.seekg (0, std::ios::beg);
    file.read (data.get(), size);
    if (!file) 
    {
        std::cerr << "error read [" << filename << "]" << std::endl;
        return nullptr;
    }
    TF_Buffer* buffer = TF_NewBufferFromString(data.get(), size);
    file.close();
    return buffer;
}

std::vector<std::string> Session::get_operations() const 
{
    std::vector<std::string> result;
    size_t pos = 0;
    TF_Operation* oper;
    while ((oper = TF_GraphNextOperation(this->graph, &pos)) != nullptr) result.emplace_back(TF_OperationName(oper));
    return result;
}

void Session::run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) 
{
    this->error_check(std::all_of(inputs.begin(), inputs.end(), [](const Tensor* i){return i->flag == 1;}),  "invalid input tensors");
    this->error_check(std::all_of(outputs.begin(), outputs.end(), [](const Tensor* o){return o->flag != -1;}), "invalid output tensors");
    std::for_each(outputs.begin(), outputs.end(), [](Tensor* o){o->clean();});
    std::vector<TF_Output> io(inputs.size());
    std::transform(inputs.begin(), inputs.end(), io.begin(), [](const Tensor* i) {return i->op;});
    std::vector<TF_Tensor*> iv(inputs.size());
    std::transform(inputs.begin(), inputs.end(), iv.begin(), [](const Tensor* i) {return i->val;});
    std::vector<TF_Output> oo(outputs.size());
    std::transform(outputs.begin(), outputs.end(), oo.begin(), [](const Tensor* o) {return o->op;});
    auto ov = new TF_Tensor*[outputs.size()];
    TF_SessionRun(this->session, nullptr, io.data(), iv.data(), inputs.size(), oo.data(), ov, outputs.size(), nullptr, 0, nullptr, this->status);
    this->status_check(true);
    for (std::size_t i=0; i<outputs.size(); i++) 
    {
        outputs[i]->val = ov[i];
        outputs[i]->flag = 1;
        outputs[i]->deduce_shape();
    }
    std::for_each(inputs.begin(), inputs.end(), [] (Tensor* i) {i->clean();});
    delete[] ov;
}

void Session::run(Tensor &input, Tensor &output) 
{
    this->run(&input, &output);
}

void Session::run(const std::vector<Tensor*> &inputs, Tensor &output) 
{
    this->run(inputs, &output);
}

void Session::run(Tensor &input, const std::vector<Tensor*> &outputs) 
{
    this->run(&input, outputs);
}

void Session::run(Tensor *input, Tensor *output) 
{
    this->run(std::vector<Tensor*>({input}), std::vector<Tensor*>({output}));
}

void Session::run(const std::vector<Tensor*> &inputs, Tensor *output) 
{
    this->run(inputs, std::vector<Tensor*>({output}));
}

void Session::run(Tensor *input, const std::vector<Tensor*> &outputs) 
{
    this->run(std::vector<Tensor*>({input}), outputs);
}

bool Session::status_check(bool throw_exc) const 
{
    if (TF_GetCode(this->status) == TF_OK) return true; 
    if (throw_exc) throw std::runtime_error(TF_Message(status));
    return false;
}

void Session::error_check(bool condition, const std::string &error) const 
{
    if (!condition) throw std::runtime_error(error);
}

}
