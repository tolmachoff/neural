#pragma once

#include "i_neural.h"

struct Lesson
{
    ublas::vector<double> x;
    ublas::vector<double> y;
};

class Teacher
{
public:
    Teacher(INeural& neural);
    ~Teacher();

    void add_lesson(const Lesson& lesson);
    int get_count() const;
    void teach(int repeats, bool to_shuffle = false);

private:
    struct Impl;
    Impl* d;

};
