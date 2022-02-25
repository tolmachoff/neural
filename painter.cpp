#include "painter.h"
#include "BMP.h"

#include <thread>
#include <mutex>

using namespace std;

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

void Painter::paint(INeural& neural, const string& filename, int size)
{
    BMP bmp(size, size);

    ublas::vector<double> n(2);
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            n(0) = static_cast<double>(i) / size;
            n(1) = static_cast<double>(j) / size;
            auto o = neural.forward(n);
            bmp.set_pixel(i, j, 0, 0, to_color(o(0)), 255);
        }
    }

    bmp.write(filename.c_str());
}
