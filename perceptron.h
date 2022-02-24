#pragma once

#include <boost/numeric/ublas/matrix.hpp>

using boost::numeric::ublas::matrix;

class Perceptron
{
public:
    Perceptron(int I, int J, int K);

    matrix<double> forward(const matrix<double>& x) const;
    void learn(const matrix<double>& x, const matrix<double>& y);

private:
    const int I;
    const int J;
    const int K;
    const double alpha;

    struct Impl;
    Impl* d;

};
