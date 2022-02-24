#pragma once

#include <boost/numeric/ublas/matrix.hpp>

using std::string;
using boost::numeric::ublas::matrix;

class INeural
{
public:
    virtual ~INeural() = default;

    virtual matrix<double> forward(const matrix<double>& x) const = 0;
    virtual void learn(const matrix<double>& x, const matrix<double>& y) = 0;

    virtual void save(const string& filename) const = 0;
    virtual void load(const string& filename) = 0;

};
