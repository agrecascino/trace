#ifndef VECTORUTIL_H
#define VECTORUTIL_H
#include <glm/glm.hpp>
void printVec3(glm::vec3 _v1) {
    std::cout << "(" << _v1.x << ", " << _v1.y << ", " << _v1.z << ")" << std::endl;
}

std::string StringifyVec3(glm::vec3 _v1) {
    return "(" + std::to_string(_v1.x) + ", " + std::to_string(_v1.y) + ", " +  std::to_string(_v1.z) +")";
}
#endif // VECTORUTIL_H
