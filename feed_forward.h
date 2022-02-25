#pragma once

#include "i_neural.h"

class FF : public INeural
{
public:
    FF(int I, int J, int K);
    ~FF() override;

    vector<int> get_sizes() const override;

    ublas::vector<double> forward(const ublas::vector<double>& x) const override;
    void learn(const ublas::vector<double>& x, const ublas::vector<double>& y) override;

    void save(const string& filename) const override;
    void load(const string& filename) override;

private:
    struct Impl;
    Impl* d;

};
