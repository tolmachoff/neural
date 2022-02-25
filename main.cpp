#include <iostream>
#include <string>
#include <random>

#include "perceptron2.h"
#include "teacher.h"
#include "painter.h"

using namespace std;
using namespace boost::numeric::ublas;

int main()
{
    Perceptron2 perceptron(2, 5, 5, 1);

    Teacher teacher(perceptron);

    default_random_engine eng;
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < 100; ++i)
    {
        matrix<double> n(1, 2);
        n(0) = dist(eng);
        n(1) = dist(eng);

        matrix<double> d(1, 1);
        if (pow(n(0) - 0.5, 2) + pow(n(1) - 0.5, 2) < 0.2)
        {
            d(0) = 1.0;
        }
        else
        {
            d(0) = 0.0;
        }

        teacher.add_lesson({n, d});
    }

    teacher.teach(1000);

    // perceptron.save("shit.txt");

    Painter::paint(perceptron, "out.bmp");

    return 0;
}
