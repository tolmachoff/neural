#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include "perceptron.h"
#include "teacher.h"

using namespace std;
using namespace boost::numeric::ublas;

void teach(INeural& neural)
{
    Teacher teacher(neural);

    ifstream in("../datasets/lib_MNIST.txt");
    assert(in);

    string dataset_name;
    getline(in, dataset_name);
    cout << "Dataset name: " << dataset_name << endl;

    while (true)
    {
        matrix<double> y(1, 10);
        int ans;
        if (!(in >> ans))
        {
            break;
        }
        y(ans) = 1.0;

        matrix<double> x(1, 784);
        for (double& val : x.data())
        {
            in >> val; 
        }

        teacher.add_lesson({x, y});
    }

    cout << "Read " << teacher.get_count() << " samples" << endl;

    teacher.teach(1);

    neural.save("shit.txt");
}

void test(INeural& neural)
{
    neural.load("shit.txt");

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
        matrix<double> d(1, 10);
        d(ans) = 1.0;
        
        matrix<double> x(1, 784);
        for (double& val : x.data())
        {
            in >> val; 
        }

        matrix<double> y = neural.forward(x);

        matrix<double> eps = d - y;
        auto it = find_if(eps.data().begin(), 
                          eps.data().end(), 
                          [](double x)
                            {
                                return abs(x) > 0.5;
                            });
        ++total_count;
        if (it == eps.data().end())
        {
            ++right_count;
        }
    }

    cout << "Tested at " << total_count << " samples" << endl;
    cout << right_count << " answers is right!" << endl;
}

int main()
{
    Perceptron perceptron(784, 256, 10);

    // teach(perceptron);
    test(perceptron);

    return 0;
}
