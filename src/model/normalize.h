#pragma once

#include "model/digits.h"

namespace game {

// Puts a drawn digit in the form MNIST digits have: the digit is scaled, keeping its proportions, so that its longer side
// is 20 pixels, and placed so that its centre of mass is the centre of the 28x28 frame (moved only as far as needed to
// keep it inside). A blank image is returned as it is.
[[nodiscard]] Pixels normalize_digit(const Pixels& image);

}
