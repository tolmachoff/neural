#include "sigmoid.h"

#include <cmath>

double f(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double f_(double x)
{
    return f(x) * (1.0 - f(x));
}