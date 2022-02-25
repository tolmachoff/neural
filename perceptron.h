#pragma once

#include "i_neural.h"

class Perceptron : public INeural
{
public:
    Perceptron(int I, int J, int K);
    ~Perceptron() override;

    ublas::vector<double> forward(const ublas::vector<double>& x) const override;
    void learn(const ublas::vector<double>& x, const ublas::vector<double>& y) override;

    void save(const string& filename) const override;
    void load(const string& filename) override;

private:
    const int I;
    const int J;
    const int K;
    const double alpha;

    struct Impl;
    Impl* d;

};
