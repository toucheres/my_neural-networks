#include "FCNN.h"

void FCNN::initRandom()
{
    std::default_random_engine engine;
    std::uniform_real_distribution<data> getnum(0.1, 0.9); // 左闭右闭区间
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
    for (size_t i = 0; i < this->numOfLays-2; i++)
    {
        hiddenlay[i]->forward();
    }
    this->outputlay->forward();
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

HiddenLay::HiddenLay(HiddenLay *last, size_t size):input(last->actived)
{
    // weight bias
    // sum actived
    // loss delweight delbias
    lastInput = nullptr;
    lastHidden = last;
    //input = lastHidden->actived;

    res_arg = lastHidden->end_arg;
    res_forward = lastHidden->end_forward;
    res_backward = lastHidden->end_backward;

    thisNodeNum = size;
    lastNodeNum = lastHidden->thisNodeNum;

    this->num_bias = this->thisNodeNum;
    this->num_weights = this->thisNodeNum * this->lastNodeNum;

    weights = this->res_arg;
    bias = this->weights + this->num_weights;

    loss = this->res_backward;
    del_weights = this->loss + this->num_bias;
    del_bias = this->weights + this->num_weights;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->del_bias + this->num_bias;
}

HiddenLay::HiddenLay(InputLay *last, size_t size):input(last->output)
{
    // weight bias
    // sum actived
    // loss delweight delbias

    lastHidden = nullptr;
    lastInput = last;
    //input = lastInput->output;

    res_arg = lastInput->res_arg;
    res_forward = lastInput->res_forward;
    res_backward = lastInput->res_backward;

    thisNodeNum = size;
    lastNodeNum = lastInput->thisNodeNum;

    this->num_bias = this->thisNodeNum;
    this->num_weights = this->thisNodeNum * this->lastNodeNum;

    weights = this->res_arg;
    bias = this->weights + this->num_weights;

    loss = this->res_backward;
    del_weights = this->loss + this->num_bias;
    del_bias = this->weights + this->num_weights;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->del_bias + this->num_bias;
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

    loss = this->res_backward;
    del_weights = this->loss + this->num_bias;
    del_bias = this->weights + this->num_weights;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->del_bias + this->num_bias;
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

    loss = this->res_backward;
    del_weights = this->loss + this->num_bias;
    del_bias = this->weights + this->num_weights;

    sum = this->res_forward;
    actived = this->sum + this->num_bias;

    end_arg = this->bias + this->num_bias;
    end_forward = this->actived + this->num_bias;
    end_backward = this->del_bias + this->num_bias;
}

void OutputLay::forward()
{
    //求和
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

argFCNN::argFCNN()
{
    this->arg = (size_t *)malloc(sizeof(size_t) * 16);
}