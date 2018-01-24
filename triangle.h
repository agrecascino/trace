#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "structs.h"

class Triangle : public Intersectable {
    friend class MeshBuilder;
public:
    Triangle(glm::vec3 pts[3], Material mat);

    virtual glm::vec3 getNormal(Ray &r, float &t);

    virtual Intersection intersect(Ray &r);

    float* getVertexBuffer();

    Material getMaterial();

    /*virtual DeviceIntersectable* returnDeviceIntersectable() {
        //return new DeviceTriangle(pts, color);
    }*/

private:
    Material mat;
    glm::vec3 pts[3];
    glm::vec3 normal;
};

#endif // TRIANGLE_H
