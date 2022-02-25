#include "perceptron.h"
#include "sigmoid.h"

#include <algorithm>
#include <random>
#include <fstream>
#include <cmath>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace std::placeholders;
using namespace boost::numeric;
using namespace boost::archive;

struct Perceptron::Impl
{
    const int I;
    const int J;
    const int K;

    const double alpha;

    ublas::matrix<double> w0;
    ublas::vector<double> b0;
    ublas::matrix<double> w1;
    ublas::vector<double> b1;

    default_random_engine eng;
    uniform_real_distribution<double> dist;


    Impl(int I, int J, int K)
        : I(I)
        , J(J)
        , K(K)
        , alpha(0.15)
        , w0(I, J)
        , b0(J)
        , w1(J, K)
        , b1(K)
        , dist(-0.1, 0.1)
    {
        randomize(w0);
        randomize(b0);
        randomize(w1);
        randomize(b1);
    }

    template <typename Container>
    void randomize(Container& m)
    {
        generate(m.data().begin(), m.data().end(), [&]() { return dist(eng); });
    }

};

template <typename Func>
ublas::vector<double> apply_func(Func func, const ublas::vector<double>& x)
{
    ublas::vector<double> y(x.size());
    transform(x.cbegin(), x.cend(), y.begin(), bind(func, _1));
    return y;
}

Perceptron::Perceptron(int I, int J, int K) : d(new Impl(I, J, K)) {}

Perceptron::~Perceptron()
{
    delete d;
}

vector<int> Perceptron::get_sizes() const
{
    return {d->I, d->J, d->K};
}

ublas::vector<double> Perceptron::forward(const ublas::vector<double>& x) const
{
    ublas::vector<double> h_ = prod(x, d->w0) + d->b0;
    ublas::vector<double> h = apply_func(f, h_);
    ublas::vector<double> o_ = prod(h, d->w1) + d->b1;
    ublas::vector<double> o = apply_func(f, o_);
    return o;
}

void Perceptron::learn(const ublas::vector<double>& x, const ublas::vector<double>& y)
{
    ublas::vector<double> h_ = prod(x, d->w0) + d->b0;
    ublas::vector<double> h = apply_func(f, h_);
    ublas::vector<double> o_ = prod(h, d->w1) + d->b1;
    ublas::vector<double> o = apply_func(f, o_);
    ublas::vector<double> eps = y - o;
    ublas::vector<double> f_o_ = apply_func(f_, o_);
    ublas::vector<double> delta1 = -2.0 * element_prod(eps, f_o_);
    ublas::matrix<double> gamma1 = outer_prod(h, delta1);
    ublas::vector<double> f_h_ = apply_func(f_, h_);
    ublas::vector<double> delta0 = element_prod(f_h_, prod(delta1, trans(d->w1)));
    ublas::matrix<double> gamma0 = outer_prod(x, delta0);

    d->w1 -= d->alpha * gamma1;
    d->b1 -= d->alpha * delta1;
    d->w0 -= d->alpha * gamma0;
    d->b0 -= d->alpha * delta0;
}

void Perceptron::save(const string& filename) const
{
    ofstream out(filename);
    assert(out);

    text_oarchive oa(out);
    oa << d->w0 << d->b0 << d->w1 << d->b1;
}

void Perceptron::load(const string& filename)
{
    ifstream in(filename);
    assert(in);

    text_iarchive ia(in);
    ia >> d->w0 >> d->b0 >> d->w1 >> d->b1;
}
