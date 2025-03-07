#include <iostream>
#include <vector>
#include <math.h>
#include <iostream>
#include <ctime>
#include <random>
#include <fstream>
#include <cassert>

#if !defined(max)
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif // max
#define private public
using data = double;

inline data ReLU_plus(data in) { return max(in, in * 0.05); }
inline data dReLU_plusdx(data in) { return in > 0 ? 1 : 0.05; }

class InputLay;
class HiddenLay;
class OutputLay;

enum 
{
    defult_learningtype,
    static_learningtype,
};

enum
{
    ordered,
    unordered,
};

struct argFCNN
{
    argFCNN();
    size_t *arg;
    size_t numOfLay;
    double learningRate;
    int learningRateType;
    size_t numofarg();
};

class FCNN
{
    friend InputLay;
    friend HiddenLay;
    friend OutputLay;
    // weight bias
    // sum actived
    // delweight delbias loss 
private:
    size_t numOfLays;
    InputLay *inputlay;
    HiddenLay **hiddenlay;
    OutputLay *outputlay;

    data *res_arg;
    data *res_forward;
    data *res_backword;

    size_t size_arg;
    size_t size_forward;
    size_t size_backward;    

    argFCNN setting;
    double learningRate;
    double loss;
    void initRandom();
public:
    data *result;
    void changeSoure(data* in);
    void forward();
    void backward(int lable);
    void backward(data* lable);
    FCNN(const argFCNN& initarg);
    ~FCNN();
};

class InputLay
{

    friend HiddenLay;
    friend OutputLay;
    friend FCNN;

public:
    InputLay(data *_res_in,data *_res_arg, data *_res_forward, data *_res_backward, size_t _size);

private:    
    data *output;
    
    data *res_arg;
    data *res_forward;
    data *res_backward;

    size_t thisNodeNum;
};
// weight bias
// sum actived
// delweight delbias loss
class HiddenLay
{
    friend HiddenLay;
    friend OutputLay;
    friend FCNN;

private:
    data *&input;

    data *res_arg;
    data *res_forward;
    data *res_backward;

    data *weights;
    data *bias;

    data *loss;
    data *del_weights;
    data *del_bias;

    data *sum;
    data *actived;

    data *end_arg;
    data *end_forward;
    data *end_backward;

    size_t thisNodeNum;
    size_t lastNodeNum;
    size_t num_bias;
    size_t num_weights;

    HiddenLay *lastHidden=nullptr;
    InputLay *lastInput=nullptr;

    data (*activefun)(data)=ReLU_plus;
    data (*dactivefun)(data)=dReLU_plusdx;

public:
    HiddenLay(HiddenLay *last, size_t size);
    HiddenLay(InputLay *last, size_t size);
    ~HiddenLay()=default;
    void forward();
    void backward();
};

class OutputLay
{
    friend HiddenLay;
    friend OutputLay;
    friend FCNN;

private:
    data *input;

    data *res_arg;
    data *res_forward;
    data *res_backward;

    data *weights;
    data *bias;

    data *loss;
    data *del_weights;
    data *del_bias;

    data *sum;
    data *actived;

    data *end_arg;
    data *end_forward;
    data *end_backward;

    size_t thisNodeNum;
    size_t lastNodeNum;
    size_t num_bias;
    size_t num_weights;

    HiddenLay *lastHidden = nullptr;
    InputLay *lastInput = nullptr;

public:
    OutputLay(HiddenLay *last, size_t size);
    OutputLay(InputLay *last, size_t size);
    ~OutputLay()=default;
    void forward();
    void backward(int lable);
    void backward(data* lable);
};

class fold_num
{
public:
    std::vector<std::vector<data>> nums_informathion_pixmap_bit;
    int lable;
    fold_num(int arg_lable);
    fold_num()=default;
};

class fileIn
{
private:
    std::vector<fold_num> res_train;
    std::vector<fold_num> res_test;
    size_t sizeOfEach_informathion_pixmap_bit;
    int type;
    bool isFileExists_ifstream(std::string &name);
    // 通过stat结构体 获得文件大小，单位字节
    size_t getFileSize1(const char *fileName);
    size_t weight;
    size_t hight;

public:
    fileIn(char*path="");
    int getlast(char *&origin_in);
};

