#include "perceptron2.h"
#include "sigmoid.h"

#include <algorithm>
#include <random>
#include <fstream>
#include <cmath>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace std::placeholders;
using namespace boost::numeric::ublas;
using namespace boost::archive;

struct Perceptron2::Impl
{
    matrix<double> w0;
    matrix<double> b0;
    matrix<double> w1;
    matrix<double> b1;
    matrix<double> w2;
    matrix<double> b2;

    default_random_engine eng;
    uniform_real_distribution<double> dist;


    Impl(int I, int J, int K, int L)
        : w0(I, J)
        , b0(1, J)
        , w1(J, K)
        , b1(1, K)
        , w2(K, L)
        , b2(1, L)
        , dist(-0.1, 0.1)
    {
        randomize_matrix(w0);
        randomize_matrix(b0);
        randomize_matrix(w1);
        randomize_matrix(b1);
        randomize_matrix(w2);
        randomize_matrix(b2);
    }

    void randomize_matrix(matrix<double>& m)
    {
        generate(m.data().begin(), m.data().end(), [&]() { return dist(eng); });
    }

};

template <typename Func>
matrix<double> apply_func(Func func, const matrix<double>& x)
{
    matrix<double> y(x.size1(), x.size2());
    transform(x.data().cbegin(), x.data().cend(), y.data().begin(), bind(func, _1));
    return y;
}

Perceptron2::Perceptron2(int I, int J, int K, int L)
    : I(I)
    , J(J)
    , K(K)
    , L(L)
    , alpha(0.15)
    , d(new Impl(I, J, K, L))
{
}

Perceptron2::~Perceptron2()
{
    delete d;
}

matrix<double> Perceptron2::forward(const matrix<double>& x) const
{
    matrix<double> h1_  = prod(x, d->w0) + d->b0;
    matrix<double> h1   = apply_func(f, h1_);
    matrix<double> h2_  = prod(h1, d->w1) + d->b1;
    matrix<double> h2   = apply_func(f, h2_);
    matrix<double> o_   = prod(h2, d->w2) + d->b2;
    matrix<double> o    = apply_func(f, o_);
    return o;
}

void Perceptron2::learn(const matrix<double>& x, const matrix<double>& y)
{
    matrix<double> h1_      = prod(x, d->w0) + d->b0;
    matrix<double> h1       = apply_func(f, h1_);
    matrix<double> h2_      = prod(h1, d->w1) + d->b1;
    matrix<double> h2       = apply_func(f, h2_);
    matrix<double> o_       = prod(h2, d->w2) + d->b2;
    matrix<double> o        = apply_func(f, o_);
    matrix<double> eps      = y - o;
    matrix<double> f_o_     = apply_func(f_, o_);
    matrix<double> delta2   = -2.0 * element_prod(eps, f_o_);
    matrix<double> gamma2   = prod(trans(h2), delta2);
    matrix<double> f_h2_    = apply_func(f_, h2_);
    matrix<double> delta1   = element_prod(f_h2_, prod(delta2, trans(d->w2)));
    matrix<double> gamma1   = prod(trans(h1), delta1);
    matrix<double> f_h1_    = apply_func(f_, h1_);
    matrix<double> delta0   = element_prod(f_h1_, prod(delta1, trans(d->w1)));
    matrix<double> gamma0   = prod(trans(x), delta0);

    d->w2 -= alpha * gamma2;
    d->b2 -= alpha * delta2;
    d->w1 -= alpha * gamma1;
    d->b1 -= alpha * delta1;
    d->w0 -= alpha * gamma0;
    d->b0 -= alpha * delta0;
}

void Perceptron2::save(const string& filename) const
{
    ofstream out(filename);
    assert(out);

    text_oarchive oa(out);
    oa << d->w0 << d->b0 << d->w1 << d->b1 << d->w2 << d->b2;
}

void Perceptron2::load(const string& filename)
{
    ifstream in(filename);
    assert(in);

    text_iarchive ia(in);
    ia >> d->w0 >> d->b0 >> d->w1 >> d->b1 >> d->w2 >> d->b2;
}
