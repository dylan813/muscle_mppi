#pragma once

// Reads a whole NumPy .npy file (format version 1.0, numpy.lib.format) of
// float64 values in C order and this machine's byte order: what NpyWriter
// writes with NpyDtype::Float64, and what np.save writes for a float64 array.
// Anything else throws rather than being misread.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyArray {
    std::vector<int>    shape;
    std::vector<double> values;   // C order: the last index varies fastest
};

inline NpyArray read_npy(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("read_npy: cannot open " + path);

    char preamble[10] = {};   // magic "\x93NUMPY" (6) + version (2) + header length (2)
    in.read(preamble, sizeof(preamble));
    if (!in || std::memcmp(preamble, "\x93NUMPY\x01\x00", 8) != 0)
        throw std::runtime_error("read_npy: not a version 1.0 .npy file: " + path);
    const size_t length = static_cast<uint8_t>(preamble[8]) | (static_cast<uint8_t>(preamble[9]) << 8);
    std::string header(length, '\0');
    in.read(header.data(), static_cast<std::streamsize>(length));

    const uint16_t probe = 1;
    const char order = *reinterpret_cast<const uint8_t*>(&probe) == 1 ? '<' : '>';
    if (header.find(std::string("'descr': '") + order + "f8'") == std::string::npos)
        throw std::runtime_error("read_npy: not float64 in this machine's byte order: " + path);
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("read_npy: not in C order: " + path);

    // The shape tuple: "(1504, 40, 24)", or "(37,)" for one dimension.
    NpyArray array;
    const size_t open  = header.find('(', header.find("'shape':"));
    const size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos)
        throw std::runtime_error("read_npy: no shape in the header: " + path);
    size_t count = 1;
    for (size_t at = open + 1; at < close;) {
        const size_t comma = std::min(header.find(',', at), close);
        const std::string item = header.substr(at, comma - at);
        if (item.find_first_not_of(' ') != std::string::npos) {
            array.shape.push_back(std::stoi(item));
            count *= static_cast<size_t>(array.shape.back());
        }
        at = comma + 1;
    }

    array.values.resize(count);
    in.read(reinterpret_cast<char*>(array.values.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (static_cast<size_t>(in.gcount()) != count * sizeof(double))
        throw std::runtime_error("read_npy: fewer values than the shape declares: " + path);
    return array;
}
