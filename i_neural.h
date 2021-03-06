#pragma once

#include <vector>
#include <boost/numeric/ublas/vector.hpp>

using std::vector;
using std::string;
using namespace boost::numeric;

class INeural
{
public:
    virtual ~INeural() = default;

    virtual vector<size_t> get_sizes() const = 0;

    virtual ublas::vector<double> predict(const ublas::vector<double>& x) const = 0;
    virtual void fit(const ublas::vector<double>& x, const ublas::vector<double>& y) = 0;

    virtual void save(const string& filename) const = 0;

};
