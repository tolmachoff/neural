#include "teacher.h"

#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

using namespace std;

struct Teacher::Impl
{
    INeural& neural;
    std::vector<Lesson> lessons;
    default_random_engine gen;

    Impl(INeural& neural) : neural(neural) {}
};

Teacher::Teacher(INeural& neural) : d(new Impl(neural)) {}

Teacher::~Teacher()
{
    delete d;
}

void Teacher::add_lesson(const Lesson& lesson)
{
    d->lessons.emplace_back(lesson);
}

void Teacher::teach(int repeats, bool to_shuffle)
{
    for (int i = 0; i < repeats; ++i)
    {
        if (to_shuffle)
        {
            shuffle(d->lessons.begin(), d->lessons.end(), d->gen);
        }

        for (const Lesson& lesson : d->lessons)
        d->neural.learn(lesson.x, lesson.y);
    }
}
