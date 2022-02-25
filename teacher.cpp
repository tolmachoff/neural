#include "teacher.h"

#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>

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
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i)
    {
        if (to_shuffle)
        {
            shuffle(d->lessons.begin(), d->lessons.end(), d->gen);
        }

        for (const Lesson& lesson : d->lessons)
        d->neural.learn(lesson.x, lesson.y);
    }
    auto finish = chrono::high_resolution_clock::now();
    auto total_count = repeats * d->lessons.size();
    auto total_ms = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    auto per_one = static_cast<double>(chrono::duration_cast<chrono::microseconds>(finish - start).count()) / total_count;
    cout << "Done " << total_count << " operations in " << total_ms << " ms" << endl;
    cout << "Average operation time: " << per_one << " us" << endl;
}
