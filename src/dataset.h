#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

namespace LightNet {

class Dataset
{
public:
    Dataset(std::string filename);

    void load(std::string filename);

    bool isLoaded() const;

    void print();

private:
    bool loaded = false;

    std::vector<std::vector<double>> data;

    bool validateDataset();
};

}

#endif // DATASET_H
