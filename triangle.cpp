#include "triangle.h"
#include <string.h>
Triangle::Triangle(glm::vec3 pts[3], Material mat) : mat(mat) {
    memcpy(this->pts, pts, sizeof(glm::vec3)*3);
    glm::vec3 u = pts[1] - pts[0];
    glm::vec3 v = pts[2] - pts[0];
    normal = glm::normalize(glm::cross(u, v));
}

glm::vec3 Triangle::getNormal(Ray &r, float &t) {
    return normal;
}

Intersection Triangle::intersect(Ray &r) {
    const float eps = 0.0000001;
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    Intersection ret;
    edge1 = pts[1] - pts[0];
    edge2 = pts[2] - pts[0];
    h = glm::cross(r.direction, edge2);
    a = glm::dot(edge1, h);
    if(a > -eps && a < eps) {
        return ret;
    }
    f = 1/a;
    s = r.origin - pts[0];
    u = f * glm::dot(s, h);
    if(u < 0.0 || u > 1.0) {
        return ret;
    }
    q = glm::cross(s, edge1);
    v = f * glm::dot(r.direction, q);
    if(v < 0.0 || u+v > 1.0)
        return ret;
    ret.intersected = true;
    ret.t = f * glm::dot(edge2, q);
    if(ret.t < eps) {
        ret.intersected = false;
        return ret;
    }
    ret.point = r.origin + r.direction*ret.t;
    ret.normal = getNormal(r, ret.t);;
    ret.mat = mat;
    ret.obj = this;
    return ret;
}

float* Triangle::getVertexBuffer() {
    return (float*)pts;
}

Material Triangle::getMaterial() {
    return mat;
}

/*virtual Triangle::DeviceIntersectable* returnDeviceIntersectable() {
    //return new DeviceTriangle(pts, color);
}*/
