#pragma once

// Writes a NumPy .npy file (format version 1.0, numpy.lib.format) of float32
// rows, one append() at a time, for np.load() in learning/. File shape is
// (rows, *row_shape): row_shape {40, 24} gives (N, 40, 24).
//
// The header is fixed-length and rewritten after every row, with the row
// written first. A run killed mid-episode therefore leaves a file whose header
// counts at most the rows on disk, which np.load reads (it ignores any extra
// trailing row).

#include <climits>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class NpyWriter {
public:
    // Creates path, replacing any file there. Throws if it can't be opened.
    NpyWriter(const std::string& path, std::vector<int> row_shape)
        : path_(path), row_shape_(std::move(row_shape))
    {
        for (int d : row_shape_) row_size_ *= static_cast<size_t>(d);

        // Room for the widest possible row count, padded so the data starts
        // on a 64-byte boundary, as the format asks.
        const size_t needed = kPreamble + header_dict(LLONG_MAX).size() + 1;   // + closing '\n'
        header_size_ = (needed + 63) / 64 * 64;

        out_.open(path_, std::ios::binary | std::ios::trunc);
        if (!out_) throw std::runtime_error("NpyWriter: cannot open " + path_);
        write_header();
    }

    // Append one row of row_size() values, stored as float32.
    void append(const double* row)
    {
        buf_.resize(row_size_);
        for (size_t i = 0; i < row_size_; ++i) buf_[i] = static_cast<float>(row[i]);

        out_.seekp(0, std::ios::end);
        out_.write(reinterpret_cast<const char*>(buf_.data()), row_size_ * sizeof(float));
        out_.flush();   // the row is on disk before the header counts it
        check_written();

        ++rows_;
        write_header();
    }

    size_t    row_size() const { return row_size_; }
    long long rows()     const { return rows_; }

private:
    static constexpr size_t kPreamble = 10;   // magic "\x93NUMPY" (6) + version (2) + header length (2)

    // The header's Python dict literal, as numpy writes it. Data is in this
    // machine's byte order, which descr records.
    std::string header_dict(long long rows) const
    {
        std::string shape = "(" + std::to_string(rows);
        if (row_shape_.empty()) shape += ",";
        for (int d : row_shape_) shape += ", " + std::to_string(d);
        shape += ")";

        const uint16_t probe = 1;
        const char order = *reinterpret_cast<const uint8_t*>(&probe) == 1 ? '<' : '>';
        return std::string("{'descr': '") + order + "f4', 'fortran_order': False, 'shape': "
               + shape + ", }";
    }

    void write_header()
    {
        std::string header = header_dict(rows_);
        header.resize(header_size_ - kPreamble - 1, ' ');
        header += '\n';

        const auto len = static_cast<uint16_t>(header.size());   // stored little-endian
        const char preamble[kPreamble] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                          static_cast<char>(len & 0xff), static_cast<char>(len >> 8)};
        out_.seekp(0);
        out_.write(preamble, kPreamble);
        out_.write(header.data(), header.size());
        out_.flush();
        check_written();
    }

    void check_written()
    {
        if (!out_) throw std::runtime_error("NpyWriter: write failed: " + path_);
    }

    std::string        path_;
    std::vector<int>   row_shape_;
    size_t             row_size_    = 1;
    size_t             header_size_ = 0;   // preamble + header text: where the data starts
    long long          rows_        = 0;
    std::ofstream      out_;
    std::vector<float> buf_;
};
