#include "perceptron.h"

#include <algorithm>
#include <random>
#include <cmath>

using namespace std;
using namespace std::placeholders;
using namespace boost::numeric::ublas;

struct Perceptron::Impl
{
    matrix<double> w0;
    matrix<double> b0;
    matrix<double> w1;
    matrix<double> b1;

    default_random_engine eng;
    uniform_real_distribution<double> dist;


    Impl(int I, int J, int K)
        : w0(I, J)
        , b0(1, J)
        , w1(J, K)
        , b1(1, K)
        , dist(-0.1, 0.1)
    {
        randomize_matrix(w0);
        randomize_matrix(b0);
        randomize_matrix(w1);
        randomize_matrix(b1);
    }

    void randomize_matrix(matrix<double>& m)
    {
        generate(m.data().begin(), m.data().end(), [&]() { return dist(eng); });
    }

};

double f(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double f_(double x)
{
    return f(x) * (1.0 - f(x));
}

template <typename Func>
matrix<double> apply_func(const matrix<double>& x, Func func)
{
    matrix<double> y(x.size1(), x.size2());
    transform(x.data().cbegin(), x.data().cend(), y.data().begin(), bind(func, _1));
    return y;
}

Perceptron::Perceptron(int I, int J, int K)
    : I(I)
    , J(J)
    , K(K)
    , alpha(0.15)
    , d(new Impl(I, J, K))
{
}

Perceptron::~Perceptron()
{
    delete d;
}

matrix<double> Perceptron::forward(const matrix<double>& x) const
{
    matrix<double> h_ = prod(x, d->w0) + d->b0;
    matrix<double> h = apply_func(h_, f);
    matrix<double> o_ = prod(h, d->w1) + d->b1;
    matrix<double> o = apply_func(o_, f);
    return o;
}

void Perceptron::learn(const matrix<double>& x, const matrix<double>& y)
{
    matrix<double> h_ = prod(x, d->w0) + d->b0;
    matrix<double> h = apply_func(h_, f);
    matrix<double> o_ = prod(h, d->w1) + d->b1;
    matrix<double> o = apply_func(o_, f);
    matrix<double> eps = y - o;
    matrix<double> f_o_ = apply_func(o_, f_);
    matrix<double> delta1 = -2.0 * element_prod(eps, f_o_);
    matrix<double> gamma1 = prod(trans(h), delta1);
    matrix<double> f_h_ = apply_func(h_, f_);
    matrix<double> delta0 = element_prod(f_h_, prod(delta1, trans(d->w1)));
    matrix<double> gamma0 = prod(trans(x), delta0);

    d->w1 -= alpha * gamma1;
    d->b1 -= alpha * delta1;
    d->w0 -= alpha * gamma0;
    d->b0 -= alpha * delta0;
}
