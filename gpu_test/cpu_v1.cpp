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

struct Obstacle
{
    public:
    double _x, _y, v_x, v_y, r;

    Obstacle()
    {
        _x = fRand(-100.0, 100.0);
        _y = fRand(-100.0, 100.0);
        v_x = fRand(0.0, 5.0);
        v_y = fRand(0.0, 5.0);
        r = 1.0;
    }

};


std::vector<double> intersectTime(Obstacle a)
{
    //distance between obstacle and scooter
    double distance = sqrt(a._x * a._x + a._y * a._y);

    //path length for start and end collision
    double d_start = distance - 2.0;
    double d_end = distance + 2.0;

    //velocity of obstacle
    double velocity = sqrt(a.v_x * a.v_x + a.v_y * a.v_y);

    //time for start and end collision
    double t_start = d_start / velocity;
    double t_end = d_end / velocity;

    //store start/end time into vector
    std::vector<double> result;
    result.push_back(t_start);
    result.push_back(t_end);

    //for test output
    //printf("(%.2lf, %.2lf), v = %.2lf\n", a._x, a._y, velocity);

    return result;
}

void print(std::vector<std::vector<double> > &list)
{
    for(int i = 0; i < list.size(); i++)
    {
        printf("start_time: %.2lf | end_time: %.2lf\n", list.at(i).at(0), list.at(i).at(1));
    }
}

int main()
{
    for(int n = 0; n < 100; n++)
    {

        long double total_time = 0.0;
        clock_t e = clock();

        //sample size : 1000
        for (int r = 0; r < 1000; r++)
        {

            //clock_t t = clock();


            //srand(time(0));
            std::vector<std::vector<double> > list;
            for(int i = 0; i < (n*10); i++)
            {

                Obstacle obs;
                std::vector<double> result = intersectTime(obs);
                list.push_back(result);
            }


            //t = clock() - t;

            //print(list);


            //long double time_elapsed_s = t / CLOCKS_PER_SEC;
            // sum up each round of time
            //total_time += time_elapsed_s;


        }

        e = clock() - e;

        //print median of 1000 samples
        //printf("%d CPU time used: %.8Lf s\n", (n*10), total_time/1000);
        double time_elapsed_s = e / (double) CLOCKS_PER_SEC;
        printf("%d CPU time used: %.8lf s\n", (n*10), time_elapsed_s/1000.0);

    }


}
