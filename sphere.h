#ifndef SPHERE_H
#define SPHERE_H
#include "structs.h"
#include <embree3/rtcore.h>

class Sphere : public Intersectable {
public:
    Sphere(glm::vec3 origin, float radius, glm::vec3 color);
    glm::vec3 getNormal(Ray &ray, float &t);
    Intersection intersect(Ray &ray);

    IntersectableInternalOptimized* optimize();
    friend void sphereBoundsFunc(const struct RTCBoundsFunctionArguments* args);
    friend void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args);
    friend void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args);
private:
    glm::vec3 origin;
    float radius;
    glm::vec3 color;
};

void sphereBoundsFunc(const struct RTCBoundsFunctionArguments* args);
void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args);
void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args);

#endif // SPHERE_H