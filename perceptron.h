#pragma once

#include <boost/numeric/ublas/matrix.hpp>

using std::string;
using boost::numeric::ublas::matrix;

class Perceptron
{
public:
    Perceptron(int I, int J, int K);
    ~Perceptron();

    matrix<double> forward(const matrix<double>& x) const;
    void learn(const matrix<double>& x, const matrix<double>& y);

    void save(const string& filename) const;
    void load(const string& filename);

private:
    const int I;
    const int J;
    const int K;
    const double alpha;

    struct Impl;
    Impl* d;

};
