#include <cuda.h>
#include <cmath>
#include <iostream>
#include <random>
#include <ctime>

/**
  * generate random double with range: @fMin ~ @fMax
  */
double fRand(double fMin, double fMax)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(fMin, fMax);
    double a = dis(gen);
    return a;
}

/**
  * create balls with radius @r, coordinate (@_x, @_y), velocity vector <@v_x, @v_y>
  */
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

/**
  * @n obstacles
  * for each obstacle, return time elapsed when collison starts @t_s and ends @t_e
  * stored in @list[]
  */
__global__ void intersectTime_g(int n, Obstacle points[], double list[])
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // process each obstacle 
    for(int j = index; j < n; j += stride)
    {
        Obstacle a = points[j];
        
        //distance @d b/w obstacle and scooter
        double d =  sqrt(a._x * a._x + a._y * a._y);
        //distance travelled when collision starts @d_s and ends @d_e
        double d_s = d - 2.0;
        double d_e = d + 2.0;
        //velocity @v of obstacle
        double v = sqrt(a.v_x * a.v_x + a.v_y * a.v_y);
        //time elapsed when collision starts @t_s and ends @t_e
        double t_s = d_s / v;
        double t_e = d_e / v;
        //store in list[j]
        list[2 * j] = t_s;
        list[2 * j + 1] = t_e;

        //for test output
        //printf("GPU: (%.2lf, %.2lf), v = %.3lf, t_s = %.2lf, t_e = %.2lf\n", a._x, a._y, v, t_s, t_e);
    }
}


void intersectTime_c(int n, Obstacle points[], double list[])
{
    for(int j = 0; j < n; j++)
    {
	Obstacle a = points[j];

	//distance @d b/w obstacle and scooter
        double d = sqrt(a._x * a._x + a._y * a._y);
	//distance travelled when collision starts @d_s and ends @d_e
	double d_s = d - 2.0;
	double d_e = d + 2.0;
	//velocity @v of obstacle
	double v = sqrt(a.v_x * a.v_x + a.v_y * a.v_y);
	//time elapsed when collision starts @t_s and ends @t_e
	double t_s = d_s / v;
	double t_e = d_e / v;
	//store in list[j]
	list[2 * j] = t_s;
        list[2 * j + 1] = t_e;

	// for test output
	//printf("CPU: (%.2lf, %.2lf), v = %.3lf, t_s = %.2lf, t_e = %.2lf\n",a._x, a._y, v, t_s, t_e);
    }
}

int main()
{
    //(@n*10) obstacles
    for(int n = 0; n < 100; n++)
    {	
	double total_time_c = 0.0;
	double total_time_g = 0.0;
    	Obstacle* points_g;
    	cudaMallocManaged(&points_g, n * 10 * sizeof(Obstacle));
        double* list_g;
        cudaMallocManaged(&list_g, n * 10 * 2 * sizeof(double));

	for(int s = 0; s < 1000; s++)
	{
	    //create same set of points for both CPU and GPU
	    Obstacle * points = new Obstacle[n * 10];
	    for(int i = 0; i < n * 10; i++)
	    {
	        points[i] = Obstacle();
	    }

            //GPU    
	    //copy points to GPU
	    cudaMemcpy(points_g, points, n * 10 * sizeof(Obstacle), cudaMemcpyHostToDevice);
            //initialize list: store 2 time data for each obstacle

            //process obstacles
            int blockSize = 256;
            int numBlocks = (n * 10 + blockSize - 1) / blockSize;
            
	    //timing
	    clock_t time = clock();
            intersectTime_g<<<numBlocks, blockSize>>>(n * 10, points_g, list_g);
            cudaMemcpy(points, points_g, n * 10 * sizeof(Obstacle), cudaMemcpyDeviceToHost);
	    cudaDeviceSynchronize();
	    time = clock() - time;
	    double elapsed_g = time / (double) CLOCKS_PER_SEC;
	    total_time_g += elapsed_g;

            //CPU
            double* list_c = new double[n * 10 * 2];
       	    clock_t e = clock();
            intersectTime_c(n * 10, points, list_c);
            e = clock() - e;
            double elapsed_c = e / (double) CLOCKS_PER_SEC;
	    total_time_c += elapsed_c;
        }
	printf("%d GPU: %.8lf s   ", (n * 10), total_time_g);
	printf("CPU: %.8lf s   ", total_time_c);
        printf("%.2lf \n", total_time_c / total_time_g);
    	cudaFree(points_g);
    	cudaFree(list_g);
    }
    
}
