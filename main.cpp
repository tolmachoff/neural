#include <iostream>
#include <string>
#include <random>

#include "perceptron.h"
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

void paint_perceptron(const Perceptron& perceptron, const string& fn)
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
            auto o = perceptron.forward(n);
            bmp.set_pixel(i, j, 0, 0, to_color(o(0)), 255);
        }
    }
    bmp.write(fn.c_str());
}

struct Lesson
{
    matrix<double> x;
    matrix<double> y;
};

int main()
{
    Perceptron perceptron(2, 5, 1);

    default_random_engine eng;
    uniform_real_distribution<double> dist(0.0, 1.0);

    const int N         = 100;
    const int REP       = 1000;

    std::vector<Lesson> lessons;
    for (int i = 0; i < N; ++i)
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

        lessons.push_back({n, d});
    }

    for (int i = 0; i < REP; ++i)
    {
        for(const Lesson& lesson : lessons)
        {
            perceptron.learn(lesson.x, lesson.y);
        }
    }

    // perceptron.save("shit.txt");
    
    paint_perceptron(perceptron, "out.bmp");

    return 0;
}
