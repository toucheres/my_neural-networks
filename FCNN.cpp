#include "FCNN.h"
#include <sstream>

void FCNN::initRandom()
{
    std::default_random_engine engine;
    std::uniform_real_distribution<data> getnum(0.1,0.2); // 左闭右闭区间
    engine.seed((unsigned int)time(0));
    for (size_t i = 0; i < this->size_arg; i++)
    {
        this->res_arg[i] = getnum(engine);
    }
    memset(this->res_forward, 0, sizeof(data) * this->size_forward);
    memset(this->res_backword, 0, sizeof(data) * this->size_backward);
}

void FCNN::changeSoure(data *in)
{
    this->inputlay->output = in;
}

void FCNN::forward()
{
    for (size_t i = 0; i < this->numOfLays - 2; i++)
    {
        hiddenlay[i]->forward();
    }
    this->outputlay->forward();
}

void FCNN::backward(int lable)
{
    double tp_loss = 0;
    tp_loss = (-1) * log10(this->result[lable]);
    // for (size_t i = 0; i < 10; i++)
    // {
    //     if(lable==i)
    //     {
    //         tp_loss += (this->result[i] - 0) * (this->result[i] - 0);
    //     }
    //     else
    //     {
    //         tp_loss += (this->result[i] - 1) * (this->result[i] - 1);
    //     }
        
    // }
    

    this->loss = tp_loss;

    if (this->setting.learningRateType == defult_learningtype)
    {
        this->learningRate = pow((this->loss / this->setting.numofarg()), 0.8);
        if (this->learningRate > 0.01)
        {
            this->learningRate = 0.01;
            //printf("too high\n");
        }
    }
    else if (this->setting.learningRateType == static_learningtype)
        this->learningRate = setting.learningRate;

    this->outputlay->backward(lable);

    for (size_t i = 0; i < this->setting.numOfLay - 2; i++)
    {
        this->hiddenlay[i]->backward();
    }

    for (size_t i = 0; i < this->size_arg; i++)
    {
        this->res_arg[i] -= this->res_backword[i] * this->learningRate;
    }
}

void FCNN::backward(data *lable)
{
    double tp_loss = 0;
    for (size_t i = 0; i < this->setting.arg[this->setting.numOfLay - 1]; i++)
        tp_loss += lable[i] * log(this->result[i]);

    this->loss = tp_loss;

    if (this->setting.learningRateType == defult_learningtype)
        this->learningRate = pow((this->loss / 10), 1.7);
    else if (this->setting.learningRateType == static_learningtype)
        this->learningRate = setting.learningRate;

    this->outputlay->backward(lable);

    for (size_t i = 0; i < this->setting.arg[this->setting.numOfLay - 2]; i++)
    {
        this->hiddenlay[i]->backward();
    }

    for (size_t i = 0; i < this->size_arg; i++)
    {
        this->res_arg[i] -= this->res_backword[i] * this->learningRate;
    }
}

FCNN::FCNN(const argFCNN &initarg)
{
    size_t sum_bias = 0;
    for (int i = 1; i < initarg.numOfLay; i++)
        sum_bias += initarg.arg[i];

    size_t sum_weights = 0;
    for (int i = 1; i < initarg.numOfLay; i++)
        sum_weights += initarg.arg[i - 1] * initarg.arg[i];

    size_t sum_loss = sum_bias;
    size_t sum_delweight = sum_weights;
    size_t sum_delbias = sum_bias;

    size_t sum_sum = sum_bias;
    size_t sum_actived = sum_bias;

    this->size_arg = sum_bias + sum_weights;
    this->size_forward = sum_sum + sum_actived;
    this->size_backward = sum_delbias + sum_delweight + sum_loss;
    this->res_arg = (data *)malloc(sizeof(data) * (this->size_arg));
    this->res_forward = (data *)malloc(sizeof(data) * (this->size_forward));
    this->res_backword = (data *)malloc(sizeof(data) * (this->size_backward));

    inputlay = new InputLay(nullptr, res_arg, res_forward, res_backword, initarg.arg[0]);

    hiddenlay = new HiddenLay *[initarg.numOfLay - 2];
    for (size_t i = 0; i < initarg.numOfLay - 2; i++)
    {
        if (i == 0)
            hiddenlay[0] = new HiddenLay(this->inputlay, initarg.arg[1]);
        else
            hiddenlay[i] = new HiddenLay(this->hiddenlay[i - 1], initarg.arg[i + 1]);
    }

    if (initarg.numOfLay == 2)
        outputlay = new OutputLay(inputlay, initarg.arg[1]);
    else
        outputlay = new OutputLay(hiddenlay[initarg.numOfLay - 3], initarg.arg[initarg.numOfLay - 1]);

    this->result = this->outputlay->actived;
    this->numOfLays = initarg.numOfLay;
    this->setting = initarg;

    this->initRandom();
}

FCNN::~FCNN()
{
    free(this->res_arg);
    free(this->res_backword);
    free(this->res_forward);
    delete (this->inputlay);
    delete (this->outputlay);
    for (size_t i = 0; i < numOfLays - 2; i++)
    {
        delete (this->hiddenlay[i]);
    }
}

InputLay::InputLay(data *__res_in, data *__res_arg, data *__res_forward, data *__res_backward, size_t __size)
{
    this->output = __res_in;
    this->res_arg = __res_arg;
    this->res_forward = __res_forward;
    this->res_backward = __res_backward;
    this->thisNodeNum = __size;
}

HiddenLay::HiddenLay(HiddenLay *last, size_t size) : input(last->actived)
{
    // weight bias
    // sum actived
    // delweight delbias loss
    lastInput = nullptr;
    lastHidden = last;
    // input = lastHidden->actived;

    res_arg = lastHidden->end_arg;
    res_forward = lastHidden->end_forward;
    res_backward = lastHidden->end_backward;

    thisNodeNum = size;
    lastNodeNum = lastHidden->thisNodeNum;

    this->num_bias = this->thisNodeNum;
    this->num_weights = this->thisNodeNum * this->lastNodeNum;

    weights = this->res_arg;
    bias = this->weights + this->num_weights;

    // loss = this->res_backward;
    // del_weights = this->loss + this->num_bias;
    // del_bias = this->weights + this->num_weights;

    del_weights = this->res_backward;
    del_bias = del_weights + this->num_weights;
    loss = del_bias + this->num_bias;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->loss + this->num_bias;
}

HiddenLay::HiddenLay(InputLay *last, size_t size) : input(last->output)
{
    // weight bias
    // sum actived
    // loss delweight delbias

    lastHidden = nullptr;
    lastInput = last;
    // input = lastInput->output;

    res_arg = lastInput->res_arg;
    res_forward = lastInput->res_forward;
    res_backward = lastInput->res_backward;

    thisNodeNum = size;
    lastNodeNum = lastInput->thisNodeNum;

    this->num_bias = this->thisNodeNum;
    this->num_weights = this->thisNodeNum * this->lastNodeNum;

    weights = this->res_arg;
    bias = this->weights + this->num_weights;

    // loss = this->res_backward;
    // del_weights = this->loss + this->num_bias;
    // del_bias = this->weights + this->num_weights;

    del_weights = this->res_backward;
    del_bias = del_weights + this->num_weights;
    loss = del_bias + this->num_bias;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->loss + this->num_bias;
}

void HiddenLay::forward()
{
    // 对这层的第i个感知机
    for (size_t i = 0; i < this->thisNodeNum; i++)
    {
        this->sum[i] = 0;
        // 对上一层的第j的感知机求和
        for (size_t j = 0; j < this->lastNodeNum; j++)
        {
            this->sum[i] += this->weights[(i * this->thisNodeNum) + j] * this->input[j];
        }
        // 激活
        this->actived[i] = this->activefun(this->sum[i]);
    }
}

void HiddenLay::backward()
{
    // 处理本层hiddenlay的loss
    for (size_t i = 0; i < this->thisNodeNum; i++)
    {
        this->loss[i] *= this->dactivefun(this->sum[i]);
    }
    // 输出层权重梯度
    for (size_t i = 0; i < this->thisNodeNum; i++)
    {
        for (size_t j = 0; j < this->lastNodeNum; j++)
        {
            this->del_weights[i * this->thisNodeNum + j] = this->loss[i] * this->input[j];
        }
    }
    // 输出层偏置梯度
    for (size_t i = 0; i < this->num_bias; i++)
    {
        this->del_bias[i] = this->loss[i];
    }
    // 上一层loss预处理
    if (this->lastHidden != nullptr)
    {
        // inputlay的loss按权反分配
        for (size_t i = 0; i < this->lastNodeNum; i++)
        {
            lastHidden->loss[i] = 0;
            for (size_t j = 0; j < this->thisNodeNum; j++)
            {
                lastHidden->loss[i] += this->loss[j] * this->weights[j * this->thisNodeNum + i];
            }
        }
    }
    else if (this->lastInput != nullptr)
    {
        // 不处理inputlay
    }
}

OutputLay::OutputLay(HiddenLay *last, size_t size)
{
    // weight bias
    // sum actived
    // loss delweight delbias
    lastInput = nullptr;
    lastHidden = last;
    input = lastHidden->actived;

    res_arg = lastHidden->end_arg;
    res_forward = lastHidden->end_forward;
    res_backward = lastHidden->end_backward;

    thisNodeNum = size;
    lastNodeNum = lastHidden->thisNodeNum;

    this->num_bias = this->thisNodeNum;
    this->num_weights = this->thisNodeNum * this->lastNodeNum;

    weights = this->res_arg;
    bias = this->weights + this->num_weights;

    // loss = this->res_backward;
    // del_weights = this->loss + this->num_bias;
    // del_bias = this->weights + this->num_weights;

    del_weights = this->res_backward;
    del_bias = del_weights + this->num_weights;
    loss = del_bias + this->num_bias;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->loss + this->num_bias;
}

OutputLay::OutputLay(InputLay *last, size_t size)
{
    // weight bias
    // sum actived
    // loss delweight delbias

    lastHidden = nullptr;
    lastInput = last;
    input = lastInput->output;

    res_arg = lastInput->res_arg;
    res_forward = lastInput->res_forward;
    res_backward = lastInput->res_backward;

    thisNodeNum = size;
    lastNodeNum = lastInput->thisNodeNum;

    this->num_bias = this->thisNodeNum;
    this->num_weights = this->thisNodeNum * this->lastNodeNum;

    weights = this->res_arg;
    bias = this->weights + this->num_weights;

    // loss = this->res_backward;
    // del_weights = this->loss + this->num_bias;
    // del_bias = this->weights + this->num_weights;

    del_weights = this->res_backward;
    del_bias = del_weights + this->num_weights;
    loss = del_bias + this->num_bias;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->loss + this->num_bias;
}

void OutputLay::forward()
{
    // 求和
    for (size_t i = 0; i < this->thisNodeNum; i++)
    {
        // 对上一层的第j的感知机求和
        this->sum[i] = 0;
        for (size_t j = 0; j < this->lastNodeNum; j++)
        {
            this->sum[i] += this->weights[(i * this->thisNodeNum) + j] * this->input[j];
        }
    }
    // 激活
    // 找到最大元素以防止 exp 计算时溢出
    double maxProb = this->sum[0];
    for (int i = 1; i < this->thisNodeNum; ++i)
    {
        if (this->sum[i] > maxProb)
        {
            maxProb = this->sum[i];
        }
    }

    // 计算指数和求和
    double sumExp = 0.0;
    for (int i = 0; i < this->thisNodeNum; ++i)
    {
        this->actived[i] = exp(this->sum[i] - maxProb);
        sumExp += this->actived[i];
    }

    // 归一化指数值以得到 softmax 概率
    for (int i = 0; i < this->thisNodeNum; ++i)
    {
        this->actived[i] /= sumExp;
    }
}

void OutputLay::backward(int lable)
{
    data *tpin = (data *)malloc(sizeof(data) * 10);
    memset(tpin, 0, sizeof(data) * 10);
    tpin[lable] = 1.0;
    this->backward(tpin);
    free(tpin);
}

void OutputLay::backward(data *lable)
{
    // 输出层误差
    for (size_t i = 0; i < this->num_bias; i++)
    {
        this->loss[i] = this->actived[i] - lable[i];
    }
    // 输出层权重梯度
    for (size_t i = 0; i < this->thisNodeNum; i++)
    {
        for (size_t j = 0; j < this->lastNodeNum; j++)
        {
            this->del_weights[i * this->thisNodeNum + j] = this->loss[i] * this->input[j];
        }
    }
    // 输出层偏置梯度
    for (size_t i = 0; i < this->num_bias; i++)
    {
        this->del_bias[i] = this->loss[i];
    }
    // 上一层loss预处理
    if (this->lastHidden != nullptr)
    {
        // inputlay的loss按权反分配
        for (size_t i = 0; i < this->lastNodeNum; i++)
        {
            lastHidden->loss[i] = 0;
            for (size_t j = 0; j < this->thisNodeNum; j++)
            {
                lastHidden->loss[i] += this->loss[j] * this->weights[j * this->thisNodeNum + i];
            }
        }
    }
    else if (this->lastInput != nullptr)
    {
        // 不处理inputlay
    }
}

argFCNN::argFCNN()
{
    this->arg = (size_t *)malloc(sizeof(size_t) * 16);
}

size_t argFCNN::numofarg()
{
    size_t num = 0;
    for (size_t i = 0; i < this->numOfLay; i++)
    {
        num += this->arg[i];
    }
    for (size_t i = 1; i < this->numOfLay; i++)
    {
        num += this->arg[i] * this->arg[i - 1];
    }
    return num;
}

bool fileIn::isFileExists_ifstream(std::string &name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

size_t fileIn::getFileSize1(const char *fileName)
{

    if (fileName == NULL)
    {
        return 0;
    }

    // 这是一个存储文件(夹)信息的结构体，其中有文件大小和创建时间、访问时间、修改时间等
    struct stat statbuf;

    // 提供文件名字符串，获得文件属性结构体
    stat(fileName, &statbuf);

    // 获取文件大小
    size_t filesize = statbuf.st_size;

    return filesize;
}

fileIn::fileIn(char *source_path)
{
    std::string path = source_path;
    // train
    std::string train_path = path;
    train_path += "/Test_set/";
    for (size_t i = 0; i < 10; i++)
    {
        this->res_train.emplace_back(i);
        for (size_t j = 1; 1 ; j++)
        {
            std::vector<data>& pixmapdata = this->res_train.back().nums_informathion_pixmap_bit[j];
            std::stringstream ss;
            ss << train_path << "\\" << i << "\\" << j;
            std::fstream file;
            file.open(ss.str(), std::ios::in); // 以只读方式打开文件

            if (!file.is_open())
            {
                std::cerr << "无法打开文件: " << ss.str() << std::endl;
                break;//该num文件夹读取完毕
            }
            size_t fileSize = this->getFileSize1(ss.str().c_str());
            std::vector<char> tp;
            tp.resize(fileSize);                       // 使用 resize 分配实际大小
                                                       // 读取文件内容
            file.read(tp.data(), fileSize);            // 使用 data() 获取 char* 指针
            std::streamsize bytesRead = file.gcount(); // 实际读取的字节数

            if (bytesRead < static_cast<std::streamsize>(fileSize))
            {
                // 处理读取不足的情况（例如文件较小或读取错误）
                tp.resize(bytesRead); // 调整大小以匹配实际读取的字节数
            }

            file.close(); // 关闭文件
        }
    }
}

fold_num::fold_num(int arg_lable):lable(arg_lable)
{
}
