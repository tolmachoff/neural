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
    vector<Example> examples;
    default_random_engine gen;

    Impl(INeural& neural) : neural(neural) {}
};

Teacher::Teacher(INeural& neural) : d(new Impl(neural)) {}

Teacher::~Teacher()
{
    delete d;
}

void Teacher::add_example(const Example& example)
{
    d->examples.emplace_back(example);
}

int Teacher::get_count() const
{
    return d->examples.size();
}

void Teacher::teach(int repeats, bool to_shuffle)
{
    size_t total_count = repeats * d->examples.size();
    size_t current_count = 0;
    int percents = 0;
    int prev_percents = -1;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i)
    {
        if (to_shuffle)
        {
            shuffle(d->examples.begin(), d->examples.end(), d->gen);
        }

        for (const Example& lesson : d->examples)
        {
            d->neural.fit(lesson.x, lesson.y);

            percents = ++current_count * 100 / total_count;
            if (percents > prev_percents)
            {
                cout << "|";
                cout.flush();
                prev_percents = percents;
            }
        }
    }
    cout << endl;
    auto finish = chrono::high_resolution_clock::now();
    auto total_ms = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    auto per_one = static_cast<double>(chrono::duration_cast<chrono::microseconds>(finish - start).count()) / total_count;
    cout << "Done " << total_count << " operations in " << total_ms << " ms" << endl;
    cout << "Average operation time: " << per_one << " us" << endl;
}
