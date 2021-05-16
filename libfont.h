#ifndef LIBFONT_H
#define LIBFONT_H
#include <stdint.h>
#include <glm/common.hpp>
#include <string>
#include <glm/vec4.hpp>

uint8_t *drawText(std::string s, glm::vec4 color);

#endif // LIBFONT_H
