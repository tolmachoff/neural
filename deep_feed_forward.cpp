#include "deep_feed_forward.h"
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

template <typename Func>
ublas::vector<double> apply_func(Func func, const ublas::vector<double>& x)
{
    ublas::vector<double> y(x.size());
    transform(x.cbegin(), x.cend(), y.begin(), bind(func, _1));
    return y;
}

struct Layer
{
    ublas::matrix<double> w;
    ublas::vector<double> b;

    Layer(size_t size1, size_t size2)
        : w(size1, size2)
        , b(size2)
    {}
};

struct Forward
{
    ublas::vector<double> o_;
    ublas::vector<double> o;    
};

struct Backward
{
    ublas::vector<double> delta;
    ublas::matrix<double> gamma;
};

struct DFFData
{
    const vector<Layer>& layers;

    vector<Forward> fwd;
    vector<Backward> bkwd;


    DFFData(const vector<Layer>& layers) : layers(layers) {}

    void forward(const ublas::vector<double>& x)
    {
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (i == 0)
            {
                ublas::vector<double> o_ = prod(x, layers[i].w) + layers[i].b;
                ublas::vector<double> o = apply_func(f, o_);
                fwd.emplace_back(Forward {move(o_), move(o)});
            }
            else
            {
                ublas::vector<double> o_ = prod(fwd.back().o, layers[i].w) + layers[i].b;
                ublas::vector<double> o = apply_func(f, o_);
                fwd.emplace_back(Forward {move(o_), move(o)});
            }
        }
    }

    void backward(const ublas::vector<double>& x,
                  const ublas::vector<double>& y)
    {
        for (size_t bi = 0; bi < layers.size(); ++bi)
        {
            size_t i = layers.size() - bi - 1;
            if (i == layers.size() - 1)
            {
                ublas::vector<double> eps = y - fwd[i].o;
                ublas::vector<double> f_o_ = apply_func(f_, fwd[i].o_);
                ublas::vector<double> delta = -2.0 * element_prod(eps, f_o_);
                ublas::matrix<double> gamma = outer_prod(fwd[i - 1].o, delta);
                bkwd.emplace_back(Backward {move(delta), move(gamma)});
            }
            else if (i > 0)
            {
                ublas::vector<double> f_o_ = apply_func(f_, fwd[i].o_);
                ublas::vector<double> tmp = prod(bkwd.back().delta, trans(layers[i + 1].w));
                ublas::vector<double> delta = element_prod(f_o_, tmp);
                ublas::matrix<double> gamma = outer_prod(fwd[i - 1].o, delta);
                bkwd.emplace_back(Backward {move(delta), move(gamma)});
            }
            else
            {
                ublas::vector<double> f_o_ = apply_func(f_, fwd[i].o_);
                ublas::vector<double> tmp = prod(bkwd.back().delta, trans(layers[i + 1].w));
                ublas::vector<double> delta = element_prod(f_o_, tmp);
                ublas::matrix<double> gamma = outer_prod(x, delta);
                bkwd.emplace_back(Backward {move(delta), move(gamma)});
            }
        }
    }
};

struct DFF::Impl
{
    vector<size_t> sizes;
    vector<Layer> layers;

    default_random_engine eng;
    uniform_real_distribution<double> dist;


    Impl(const vector<size_t>& sizes)
        : sizes(sizes)
        , dist(-0.1, 0.1)
    {
        for(size_t i = 0; i < sizes.size() - 1; ++i)
        {
            layers.emplace_back(Layer(sizes[i], sizes[i + 1]));
            randomize(layers.back().w);
            randomize(layers.back().b);
        }
    }

    Impl() = default;

    template <typename Container>
    void randomize(Container& m)
    {
        generate(m.data().begin(), m.data().end(), [&]() { return dist(eng); });
    }

};


DFF::DFF(initializer_list<size_t> sizes) : d(new Impl(sizes)) {}

DFF::DFF(const string& filename) : d(new Impl)
{
    ifstream in(filename);
    assert(in);
    size_t N;
    in >> N;
    d->sizes.resize(N);
    for (auto& size : d->sizes)
    {
        in >> size;
    }

    text_iarchive ia(in);
    for(size_t i = 0; i < d->sizes.size() - 1; ++i)
    {
        d->layers.emplace_back(Layer(d->sizes[i], d->sizes[i + 1]));
        ia >> d->layers.back().w >> d->layers.back().b;
    }
}

DFF::~DFF()
{
    delete d;
}

vector<size_t> DFF::get_sizes() const
{
    return d->sizes;
}

ublas::vector<double> DFF::predict(const ublas::vector<double>& x) const
{
    DFFData data(d->layers);
    data.forward(x);
    return data.fwd.back().o;
}

void DFF::fit(const ublas::vector<double>& x, const ublas::vector<double>& y)
{
    DFFData data(d->layers);
    data.forward(x);
    data.backward(x, y);

    const double alpha = 0.15;

    for (int i = 0; i < d->layers.size(); ++i)
    {
        size_t bkwd_i = d->layers.size() - i - 1;
        d->layers[i].w -= alpha * data.bkwd[bkwd_i].gamma;
        d->layers[i].b -= alpha * data.bkwd[bkwd_i].delta;
    }
}

void DFF::save(const string& filename) const
{
    ofstream out(filename);
    assert(out);
    out << d->sizes.size() << " ";
    for (size_t size : d->sizes)
    {
        out << size << " ";
    }
    out << endl;

    text_oarchive oa(out);
    for (auto& layer : d->layers)
    {
        oa << layer.w << layer.b;
    }
}
