#include <iostream>
#include"FCNN.h"
int main(int, char**){
    argFCNN a;
    a.numOfLay = 3;
    a.arg[0] = 10;
    a.arg[1] = 11;
    a.arg[2] = 5;
    FCNN b(a);

    data *in = (data *)malloc(sizeof(data) * 10);

    std::default_random_engine engine;
    std::uniform_real_distribution<data> getnum(0.001, 0.999); // 左闭右闭区间
    engine.seed(time(0));
    for (size_t i = 0; i < 10; i++)
    {
        in[i] = getnum(engine);
    }

    b.changeSoure(in);
    b.forward();
    for (size_t i = 0; i < 5; i++)
    {
        std::cout << b.result[i] << " ";
    }

    int c = 0;
}
