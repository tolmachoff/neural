#pragma once

#include "i_neural.h"

class FF : public INeural
{
public:
    FF(size_t I, size_t J, size_t K);
    FF(const string& filename);

    ~FF() override;

    vector<size_t> get_sizes() const override;

    ublas::vector<double> forward(const ublas::vector<double>& x) const override;
    void learn(const ublas::vector<double>& x, const ublas::vector<double>& y) override;

    void save(const string& filename) const override;

private:
    struct Impl;
    Impl* d;

};
