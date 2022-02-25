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

    virtual vector<int> get_sizes() const = 0;

    virtual ublas::vector<double> forward(const ublas::vector<double>& x) const = 0;
    virtual void learn(const ublas::vector<double>& x, const ublas::vector<double>& y) = 0;

    virtual void save(const string& filename) const = 0;
    virtual void load(const string& filename) = 0;

};
