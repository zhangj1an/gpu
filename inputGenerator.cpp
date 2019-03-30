#include <cmath>
#include <iostream>
#include <random>
#include <ctime>

double fRand(double fMin, double fMax)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(fMin, fMax);
    double a = dis(gen);
    return a;
}

int main()
{
    for(int i = 0; i < 1000; i++)
    {
        double _x = fRand(-100.0, 100.0);
        double _y = fRand(-100.0, 100.0);
        double r = 1.0;
        double v_x = -1 * _x * fRand(0, 5.0);
        double v_y = -1 * _y * fRand(0, 5.0);
        printf("%.8lf %.8lf %.1lf %.8lf %.8lf ", _x, _y, r, v_x, v_y);
    }
}
