#pragma once

#include <cmath>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace format_bench {
struct Case {
    std::string label, model, tensor;
    int expert=0, n=0, k=0;
    double file_bpw=0;
    std::string name() const { return label+"_"+std::to_string(n)+"x"+std::to_string(k); }
};
inline std::vector<Case> read_cases(std::istream& file) {
    std::vector<Case> result;
    std::unordered_set<std::string> names;
    for (std::string line; std::getline(file,line);) {
        if (line.empty()) continue;
        std::istringstream row(line); Case c; std::string expert,n,k,bpw;
        if (!std::getline(row,c.label,'\t') || !std::getline(row,c.model,'\t') ||
            !std::getline(row,c.tensor,'\t') || !std::getline(row,expert,'\t') ||
            !std::getline(row,n,'\t') || !std::getline(row,k,'\t') || !std::getline(row,bpw))
            throw std::runtime_error("invalid cases manifest row");
        c.expert=std::stoi(expert);c.n=std::stoi(n);c.k=std::stoi(k);c.file_bpw=std::stod(bpw);
        if (c.n<=0 || c.k<=0 || c.expert<0 || !std::isfinite(c.file_bpw))
            throw std::runtime_error("invalid case dimensions");
        if (!names.insert(c.name()).second)
            throw std::runtime_error("duplicate benchmark case key: "+c.name()+"; use a distinct label");
        result.push_back(std::move(c));
    }
    if(result.empty()) throw std::runtime_error("empty cases manifest");
    return result;
}
inline std::vector<Case> read_cases(const char* path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open cases manifest");
    return read_cases(file);
}
} // namespace format_bench
