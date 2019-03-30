#include <iostream>
#include <ctime>

string response;

struct Obstacle
{
public:
    double c_x, c_y, r, v_x, v_y;

    Obstacle(double cx, double cy, double r0, double vx, double vy)
    {
        c_x = cx;
        c_y = cy;
        r = r0;
        v_x = vx;
        v_y = vy;
    }
}

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
      	double t_s = 0;
      	double t_e = 0;
      	//Case 1: object alrd collide w scooter
      	if(d <= 1)
      	{
       	    t_s = 0;
      	    t_e = infty();
              }
      	//Case 2: object move in opposite dir w.r.t scooter
      	else if(a._x * a.v_x >= 0 || a._y * a.v_y >= 0)
      	{
            t_s = infty();
      	    t_e = infty();
      	} else
      	{
      	    double v = sqrt(a.v_x * a.v_x + a.v_y * a.v_y);
      	    double delta_t = 2 * sqrt((1.0 + a.r) * (1.0 + a.r) - 1) / v;
      	    t_s = (sqrt(d * d - 1.0) / v) - 0.5 * delta_t;
      	    t_e = t_s + delta_t;
      	}
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
        double d =  sqrt(a._x * a._x + a._y * a._y);
        double t_s = 0;
        double t_e = 0;
        //Case 1: object alrd collide w scooter
        if(d <= 1)
        {
            t_s = 0;
            t_e = infty();
        }
        //Case 2: object move in opposite dir w.r.t scooter
        else if(a._x * a.v_x >= 0 || a._y * a.v_y >= 0)
        {
            t_s = infty();
            t_e = infty();
        } else
        {
            double v = sqrt(a.v_x * a.v_x + a.v_y * a.v_y);
            double delta_t = 2 * sqrt((1.0 + a.r) * (1.0 + a.r) - 1) / v;
            t_s = (sqrt(d * d - 1.0) / v) - 0.5 * delta_t;
            t_e = t_s + delta_t;
        }
            //store in list[j]
            list[2 * j] = t_s;
            list[2 * j + 1] = t_e;

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
    bool valid = true;
    int n = 0;
    cin >> n;
    Obstacle* obstacles[n] = {};
    double* result[2 * n] = {};
    int c_x, c_y, r, v_x, v_y;
    for(int i = 0; i < n; i++)
    {
        cin >> c_x >> c_y >> r >> v_x >> v_y;
        obstacles[i] = new Obstacle(c_x, c_y, r, v_x, v_y);
    }
    while (valid)
    {
        cout << "Use GPU for computation ? (Y/N)" << endl;
        cin >> response;
        if(response == "Y" || response == "y" || response == "yes")
        {
            cout << "Use unified memory? (Y/N)" << endl;
            cin >> response;
            if(response == "Y" || response == "y" || response == "yes")
            {
                //unified gpu memory
                cout << "unified gpu memory incurred.\n";
                valid = false;
                result = gpu_unified(n, obstacles, result);
            } else if (response == "N" || response == "n" || response == "no")
            {
                //discrete gpu memory
                cout << "discrete gpu memory incurred.\n";
                valid = false;
                result = gpu_discrete(n, obstacles, result);
            } else {
                cout << "invalid input. try again\n";
            }
        } else if (response == "N" || response == "n" || response == "no")
        {
            // cpu
            cout << "cpu memory incurred.\n";
            valid = false;
            result = cpu(n, obstacles, result);
        } else {
            cout << "invalid input. try again\n";
        }
    }
    //print output
    for(int i = 0; i < n; i++)
    {
        cout << result[2 * i] << result[2 * i + 1] << endl;
    }
}
