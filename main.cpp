#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>

#include "feed_forward.h"
#include "deep_feed_forward.h"
#include "teacher.h"
#include "painter.h"

using namespace std;
using namespace boost::numeric;

void learn_circle()
{
    DFF neural {2, 5, 5, 1};

    Teacher teacher(neural);

    default_random_engine eng;
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < 100; ++i)
    {
        ublas::vector<double> n(2);
        n(0) = dist(eng);
        n(1) = dist(eng);

        ublas::vector<double> d(1);
        if (pow(n(0) - 0.5, 2) + pow(n(1) - 0.5, 2) < 0.2)
        {
            d(0) = 1.0;
        }
        else
        {
            d(0) = 0.0;
        }

        teacher.add_example({n, d});
    }

    teacher.teach(1000);

    Painter::paint(neural, "out.bmp");
}

void teach()
{
    FF neural(784, 256, 10);

    Teacher teacher(neural);

    ifstream in("../datasets/lib_MNIST.txt");
    assert(in);

    string dataset_name;
    getline(in, dataset_name);
    cout << "Dataset name: " << dataset_name << endl;

    while (true)
    {
        ublas::vector<double> y(10);
        int ans;
        if (!(in >> ans))
        {
            break;
        }
        y(ans) = 1.0;

        ublas::vector<double> x(784);
        for (double& val : x)
        {
            in >> val; 
        }

        teacher.add_example({x, y});
    }

    cout << "Read " << teacher.get_count() << " samples" << endl;

    teacher.teach(1);

    neural.save("shit.txt");
}

void test()
{
    FF neural("shit.txt");

    assert(neural.get_sizes().front() == 784);
    assert(neural.get_sizes().back() == 10);

    ifstream in("../datasets/lib_10k.txt");
    assert(in);

    string dataset_name;
    getline(in, dataset_name);
    cout << "Dataset name: " << dataset_name << endl;

    size_t total_count = 0;
    size_t right_count = 0;
    while (true)
    {
        int ans;
        if (!(in >> ans))
        {
            break;
        }
        ublas::vector<double> d(10);
        d(ans) = 1.0;
        
        ublas::vector<double> x(784);
        for (double& val : x.data())
        {
            in >> val; 
        }

        ublas::vector<double> y = neural.predict(x);

        ublas::vector<double> eps = d - y;
        auto it = find_if(eps.begin(), 
                          eps.end(), 
                            [](double x)
                            {
                                return abs(x) > 0.5;
                            });
        ++total_count;
        if (it == eps.end())
        {
            ++right_count;
        }
    }

    cout << "Tested at " << total_count << " samples" << endl;
    cout << right_count << " answers is right!" << endl;
}

int main()
{
    learn_circle();
    // teach();
    // test();

    return 0;
}
