#include <iostream>
#include <ctime>
#include <string>

struct Obstacle
{
public:
    double c_x, c_y, r, v_x, v_y;

    Obstacle()
    {
      	c_x = 0;
      	c_y = 0;
      	r = 1.0;
      	v_x = 0;
      	v_y = 0;
    }

    Obstacle(double cx, double cy, double r0, double vx, double vy)
    {
        c_x = cx;
        c_y = cy;
        r = r0;
        v_x = vx;
        v_y = vy;
    }
};

__device__ double infty_g(void)
{
    const unsigned long long ieee754inf = 0x7ff0000000000000;
    return __longlong_as_double(ieee754inf);
}

double infty_c(void)
{
    const unsigned long long ieee754inf = 0x7ff0000000000000;
    return (double)(ieee754inf);
}
__global__ void intersectTime_g(int n, Obstacle points[], double list[])
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // process each obstacle
    for(int j = index; j < n; j += stride)
    {
        Obstacle a = points[j];

        double d =  sqrt(a._x * a._x + a._y * a._y);
      	double t_s = 0;
      	double t_e = 0;
        double d_temp = max(d, 1);
        double v = sqrt(a.v_x * a.v_x + a.v_y * a.v_y);
        double delta_t = 2 * sqrt((1.0 + a.r) * (1.0 + a.r) - 1) / v;
        t_s = (sqrt(d_temp * d_temp - 1.0) / v) - 0.5 * delta_t;
        t_e = t_s + delta_t;
        double ts[] = {t_s, infty_g(), infty_g(), 0, 0, 0};
        double te[] = {t_e, infty_g(), infty_g(), infty_g(), infty_g(), infty_g()};

        int cond = 3 * (d <=1) + (a._x * a.v_x >= 0) + (a._x * a.v_x >= 0);
        //store in list[j]
        list[2 * j] = ts[cond];
        list[2 * j + 1] = te[cond];

        //for test output
        //printf("GPU: (%.2lf, %.2lf), v = %.3lf, t_s = %.2lf, t_e = %.2lf\n", a._x, a._y, v, t_s, t_e);
    }
}

void intersectTime_c(int n, Obstacle points[], double list[])
{
    for(int j = 0; j < n; j++)
    {
        Obstacle a = points[j];

        double d =  sqrt(a._x * a._x + a._y * a._y);
      	double t_s = 0;
      	double t_e = 0;
        double d_temp = max(d, 1);
        double v = sqrt(a.v_x * a.v_x + a.v_y * a.v_y);
        double delta_t = 2 * sqrt((1.0 + a.r) * (1.0 + a.r) - 1) / v;
        t_s = (sqrt(d_temp * d_temp - 1.0) / v) - 0.5 * delta_t;
        t_e = t_s + delta_t;
        double ts[] = {t_s, infty_c(), infty_c(), 0, 0, 0};
        double te[] = {t_e, infty_c(), infty_c(), infty_c(), infty_c(), infty_c()};

        int cond = 3 * (d <=1) + (a._x * a.v_x >= 0) + (a._x * a.v_x >= 0);
        //store in list[j]
        list[2 * j] = ts[cond];
        list[2 * j + 1] = te[cond];

        //for test output
        //printf("GPU: (%.2lf, %.2lf), v = %.3lf, t_s = %.2lf, t_e = %.2lf\n", a._x, a._y, v, t_s, t_e);
    }
}

double* gpu_discrete(int n, Obstacle points[], double list[])
{
    Obstacle* points_g;
    cudaMalloc(&points_g, n * sizeof(Obstacle));
    double* list_g;
    cudaMalloc(&list_g, n * 2 * sizeof(double));
    cudaMemcpy(points_g, points, n * sizeof(Obstacle), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    intersectTime_g<<<numBlocks, blockSize>>>(n, points_g, list_g);
    cudaDeviceSynchronize();
    cudaMemcpy(list, list_g, n * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    return list;
}

double* gpu_unified(int n, Obstacle points[], double list[])
{
    Obstacle* points_g;
    cudaMallocManaged(&points_g, n * sizeof(Obstacle));
    double* list_g;
    cudaMallocManaged(&list_g, n * 2 * sizeof(double));
    cudaMemcpy(points_g, points, n * sizeof(Obstacle), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    intersectTime_g<<<numBlocks, blockSize>>>(n, points_g, list_g);
    cudaDeviceSynchronize();
    return list_g;
}

double* cpu(int n, Obstacle points[], double list[])
{
    intersectTime_c(n, points, list);
    return list;
}

int main(int argc, char *argv[])
{
    std::string response;
    bool valid = false;
    int n = 0;
    std::cin >> n;
    Obstacle* obstacles = new Obstacle[n];
    double* result = new double[2 * n];
    double c_x, c_y, r, v_x, v_y;
    for(int i = 0; i < n; i++)
    {
        std::cin >> c_x >> c_y >> r >> v_x >> v_y;
        obstacles[i] = Obstacle(c_x, c_y, r, v_x, v_y);
    }
    valid = true;
    // while (valid)
    {
        std::cout << "Use GPU for computation ? (Y/N)" << std::endl;
        std::cin >> response;
        if(response == "Y" || response == "y" || response == "yes")
        {
            std::cout << "Use unified memory? (Y/N)" << std::endl;
            std::cin >> response;
            if(response == "Y" || response == "y" || response == "yes")
            {
                //unified gpu memory
                std::cout << "unified gpu memory incurred.\n";
                valid = false;
                result = gpu_unified(n, obstacles, result);
            } else if (response == "N" || response == "n" || response == "no")
            {
                //discrete gpu memory
                std::cout << "discrete gpu memory incurred.\n";
                valid = false;
                result = gpu_discrete(n, obstacles, result);
            } else {
                std::cout << "invalid input. try again\n";
            }
        } else if (response == "N" || response == "n" || response == "no")
        {
            // cpu
            std::cout << "cpu memory incurred.\n";
            valid = false;
            result = cpu(n, obstacles, result);
        } else {
            std::cout << "invalid input. try again\n";
        }
    }
    //print output
    for(int i = 0; i < n; i++)
    {
        std::cout << i << " " << result[2 * i] << " " << result[2 * i + 1] << std::endl;
    }
}
