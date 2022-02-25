#pragma once

#include "i_neural.h"

struct Lesson
{
    matrix<double> x;
    matrix<double> y;
};

class Teacher
{
public:
    Teacher(INeural& neural);
    ~Teacher();

    void add_lesson(const Lesson& lesson);
    void teach(int repeats, bool to_shuffle = false);

private:
    struct Impl;
    Impl* d;

};
