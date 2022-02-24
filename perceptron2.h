#pragma once

#include "i_neural.h"

class Perceptron2 : public INeural
{
public:
    Perceptron2(int I, int J, int K, int L);
    ~Perceptron2() override;

    matrix<double> forward(const matrix<double>& x) const override;
    void learn(const matrix<double>& x, const matrix<double>& y) override;

    void save(const string& filename) const override;
    void load(const string& filename) override;

private:
    const int I;
    const int J;
    const int K;
    const int L;
    const double alpha;

    struct Impl;
    Impl* d;

};
