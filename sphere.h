#ifndef SPHERE_H
#define SPHERE_H
#include "structs.h"
#include <embree3/rtcore.h>

class Sphere : public Intersectable {
public:
    Sphere(glm::vec3 origin, float radius, Material mat);
    glm::vec3 getNormal(Ray &ray, float &t);
    Intersection intersect(Ray &ray);

    IntersectableInternalOptimized* optimize();
    friend int main(int argc, char **argv);
    friend void sphereBoundsFunc(const struct RTCBoundsFunctionArguments* args);
    friend void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args);
    friend void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args);

    Material getMaterial();

    unsigned int getGeomID() const;

    void setGeomID(unsigned int id);
private:
    unsigned int geomID;
    glm::vec3 origin;
    float radius;
    Material mat;
};

void sphereBoundsFunc(const struct RTCBoundsFunctionArguments* args);
void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args);
void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args);

#endif // SPHERE_H
