#pragma once

#include "i_neural.h"

using std::initializer_list;

class DFF : public INeural
{
public:
    DFF(initializer_list<size_t> sizes);
    DFF(const string& filename);

    ~DFF() override;

    vector<size_t> get_sizes() const override;

    ublas::vector<double> predict(const ublas::vector<double>& x) const override;
    void fit(const ublas::vector<double>& x, const ublas::vector<double>& y) override;

    void save(const string& filename) const override;

private:
    struct Impl;
    Impl* d;

};
