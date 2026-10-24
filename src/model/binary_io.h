#pragma once

#include <bit>
#include <istream>
#include <ostream>

namespace game {

static_assert(std::endian::native == std::endian::little, "model and dataset files are little-endian");

// Reads the raw bytes of a trivially copyable value; false when the stream ran out.
template<typename T>
bool read_value(std::istream& in, T& value) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

template<typename T>
void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

}
