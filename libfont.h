#ifndef LIBFONT_H
#define LIBFONT_H
#include <stdint.h>
#include <glm/common.hpp>
#include <string>

uint8_t *drawText(std::string s, glm::vec4 color);

#endif // LIBFONT_H
