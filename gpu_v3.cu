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


__global__ double* intersectTime_g(Obstacle a)
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
    double *result;
    cudaMallocManaged(&result, 2 * sizeof(double));

    result[0] = t_start;
    result[1] = t_end;

    //for test output
    //printf("(%.2lf, %.2lf), v = %.2lf\n", a._x, a._y, velocity);

    return result;
}

std::vector<double> intersectTime_c(Obstacle a)
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
    //no of obstacles = n * 10
    for(int n = 0; n < 100; n++)
    {
        /**
          * GPU
          */

        //start timing
        float elapsed_g = 0;
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
        //sample size: 1000
        for (int r = 0; r < 1000; r++)
        {
            srand(time(0));

            /* Allocate Unified memeory - accessible from CPU or GPU */
            //points: store n * 10 obstacle object.
            Obstacle *points;
            cudaMallocManaged(&points, n * 10 * sizeof(Obstacle));
            //list : store start_time, end_time for all obstacles
            //n * 10 obstacles has 2 double time values
            double *list;
            cudaMallocManaged(&list, n * 10 * 2 * sizeof(double));

            //initialize obstacle array on the host
            for(int i = 0; i < (n * 10); i++)
            {
                Obstacle obs;
                points[i] = obs;
            }

            // run kernel on GPU
            for(int j = 0; j < (n * 10); j++)
            {
                double *result;
                cudaMallocManaged(&result, 2 * sizeof(double));

                /*Streaming Multiprocessors*/
                int blockSize = 256;
                int numBlocks = (n * 10 + blockSize - 1) / blockSize;

                result = intersectTime_g<<<numBlocks, blockSize>>>(n*10, list, points);

                //cuda code to allocate memory for vector
                list[j] = result;

                cudaFree(result);
            }

            cudaFree(points);
            cudaFree(list);

            cudaDeviceSynchronize();


        }
        //print time for gpu
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize (stop) );
        HANDLE_ERROR(cudaEventElapsedTime(&elapsed_g, start, stop) );
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
        printf("%d GPU: %.8lf s   ", (n*10), elapsed_g);



        /**
          * CPU
          */

        //start timing
        long double total_time = 0.0;
        clock_t e = clock();
        //sample size : 1000
        for (int r = 0; r < 1000; r++)
        {
            srand(time(0));
            std::vector<std::vector<double> > list;
            for(int i = 0; i < (n*10); i++)
            {
                Obstacle obs;
                std::vector<double> result = intersectTime_c(obs);
                list.push_back(result);
            }
        }
        //print time for cpu
        e = clock() - e;
        double elapsed_c = e / (double) CLOCKS_PER_SEC;
        // calculate time used for each sample
        printf("CPU: %.8lf s   ", elapsed_c / 1000.0);

        //print CPU / GPU : increase rate
        printf("%.2lf \n", elapsed_c / elapsed_g);
    }


}
