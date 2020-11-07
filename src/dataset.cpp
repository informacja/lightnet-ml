#include "dataset.h"

namespace LightNet {

Dataset::Dataset(std::string filename)
{
    load(filename);
}

void Dataset::load(std::string filename)
{
    std::fstream fin;

    fin.open(filename, std::ios::in);

    if(!fin.is_open()){
        std::cerr << " -- unable to open file at: " << filename << std::endl;
        return;
    }

    std::vector<double> row;

    std::string line, word;

    while (fin.good()) {

        row.clear();

        std::getline(fin, line);

        std::stringstream s(line);

        while(std::getline(s, word, ',')){

            if(word.empty()){
                break;
            }

            row.push_back(std::stod(word));
        }

        if(!row.empty()){
            data.push_back(row);
        }

    }

    loaded = validateDataset();
}

bool Dataset::isLoaded() const
{
    return loaded;
}

void Dataset::print()
{
    std::cout << "Dataset => rows: " << data.size() << " columns: " << data[0].size() << "\n" << std::endl;

    for(std::vector<double> &row : data){

        size_t counter = 0;

        for(double &cellData : row){

            if(counter < row.size() - 1){
                std::cout << cellData << ", ";
            }else {
                std::cout << cellData;
            }

            counter++;
        }

        std::cout << std::endl;
    }
}

bool Dataset::validateDataset()
{

    // check that the rows are of equal sizes
    size_t previousRowSize = 0;
    size_t counter = 0;

    for(std::vector<double> &row : data){

        if(counter == 0){
            // first iteration
        }else {

            if(row.size() != previousRowSize){
                std::cerr << " -- all rows must be of the same size" << std::endl
                          << "current row size: " << row.size() << std::endl
                          << "previous row size: " << previousRowSize << std::endl;
                return false;
            }
        }

        previousRowSize = row.size();
        counter++;
    }

    return true;
}

// end of namespace
}
