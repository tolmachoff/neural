#pragma once

#include "i_neural.h"

struct Example
{
    ublas::vector<double> x;
    ublas::vector<double> y;
};

class Teacher
{
public:
    Teacher(INeural& neural);
    ~Teacher();

    void add_example(const Example& example);
    int get_count() const;
    void teach(int epochs, bool to_shuffle = false);

private:
    struct Impl;
    Impl* d;

};
