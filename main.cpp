#include <iostream>
#include <string>
#include <random>

#include "perceptron2.h"
#include "teacher.h"
#include "BMP.h"

using namespace std;
using namespace boost::numeric::ublas;

uint8_t to_color(double x)
{
    double c = 255.0 * x;
    if (c < 0)
    {
        return 0;
    }
    else if (c > 255)
    {
        return 255;
    }
    else
    {
        return static_cast<uint8_t>(c);
    }
}

void paint_perceptron(const INeural& neural, const string& fn)
{
    const int IMAGE_SIZE = 500;

    BMP bmp(IMAGE_SIZE, IMAGE_SIZE);
    matrix<double> n(1, 2);
    for (int i = 0; i < IMAGE_SIZE; ++i)
    {
        for (int j = 0; j < IMAGE_SIZE; ++j)
        {
            n(0) = static_cast<double>(i) / IMAGE_SIZE;
            n(1) = static_cast<double>(j) / IMAGE_SIZE;
            auto o = neural.forward(n);
            bmp.set_pixel(i, j, 0, 0, to_color(o(0)), 255);
        }
    }
    bmp.write(fn.c_str());
}

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
    
    paint_perceptron(perceptron, "out.bmp");

    return 0;
}
