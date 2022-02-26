#include "feed_forward.h"
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

struct FF::Impl
{
    ublas::matrix<double> w0;
    ublas::vector<double> b0;
    ublas::matrix<double> w1;
    ublas::vector<double> b1;

    default_random_engine eng;
    uniform_real_distribution<double> dist;

    Impl(int I, int J, int K)
        : w0(I, J)
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

    Impl() = default;

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

FF::FF(size_t I, size_t J, size_t K) : d(new Impl(I, J, K)) {}

FF::FF(const string& filename) : d(new Impl)
{
    ifstream in(filename);
    assert(in);

    text_iarchive ia(in);
    ia >> d->w0 >> d->b0 >> d->w1 >> d->b1;
}

FF::~FF()
{
    delete d;
}

vector<size_t> FF::get_sizes() const
{
    return {d->w0.size1(), d->w0.size2(), d->w1.size2()};
}

ublas::vector<double> FF::forward(const ublas::vector<double>& x) const
{
    ublas::vector<double> h_ = prod(x, d->w0) + d->b0;
    ublas::vector<double> h = apply_func(f, h_);
    ublas::vector<double> o_ = prod(h, d->w1) + d->b1;
    ublas::vector<double> o = apply_func(f, o_);
    return o;
}

void FF::learn(const ublas::vector<double>& x, const ublas::vector<double>& y)
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

    const double alpha = 0.15;

    d->w1 -= alpha * gamma1;
    d->b1 -= alpha * delta1;
    d->w0 -= alpha * gamma0;
    d->b0 -= alpha * delta0;
}

void FF::save(const string& filename) const
{
    ofstream out(filename);
    assert(out);

    text_oarchive oa(out);
    oa << d->w0 << d->b0 << d->w1 << d->b1;
}
