#include <cmath>
#include <iostream>
#include <random>
#include <ctime>

using namespace std;
double * add(double a[]) {
    double * b;
    b[0] = a[0] + 1.0;
    cout << "success add first" << endl;
    b[1] = a[1] + 1.0;
    b[2] = a[2] + 1.0;
    return b;
}

int main()
{
    double *a = NULL;
    a = new double[3];
    a[0] = 0.0;
    a[1] = 1.0;
    a[2] = 2.0;
    cout << "array created" << endl;
    a = add(a);
    cout << a[0] << " " << a[1] << " " << a[2] << endl;
}
