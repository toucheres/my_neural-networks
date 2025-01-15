#include <iostream>
#define private public
#include "FCNN.h"
int main(int, char **)
{
    argFCNN a;
    a.numOfLay = 3;
    a.arg[0] = 10;
    a.arg[1] = 11;
    a.arg[2] = 5;
    a.learningRateType = defult_learningtype;
    FCNN b(a);

    data *in = (data *)malloc(sizeof(data) * 10);

    std::default_random_engine engine;
    std::uniform_real_distribution<data> getnum(0.001, 0.999); // 左闭右闭区间
    engine.seed((unsigned int)time(0));
    for (size_t i = 0; i < 10; i++)
    {
        in[i] = getnum(engine);
    }

    b.changeSoure(in);

    for (size_t i = 0; i < 100000; i++)
    {
        b.forward();
        // for (size_t i = 0; i < 5; i++)
        // {
        //     std::cout << b.result[i] << " ";
        // }
        b.backward(0);
        std::cout << std::endl;
        printf("%lf",b.loss);
        std::cout << std::endl;
    }

    int c = 0;
}
