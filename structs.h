#ifndef STRUCTS_H
#define STRUCTS_H
#include <glm/glm.hpp>
#include "cudapatch.h"
#include <vector>
#include <string>
struct CameraConfig {
    CameraConfig() {}
    CameraConfig(glm::vec3 center, glm::vec3 lookat,
                 glm::vec3 up) : center(center),
        lookat(lookat), up(up) {}

    glm::vec3 center;
    glm::vec3 lookat;
    glm::vec3 up;
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    size_t bounces = 0;
};

struct Framebuffer {
    size_t x;
    size_t y;
    uint8_t *fb;
};

struct Material {
    glm::vec3 color;
    bool reflective = false;
};

struct Light {
    Light(glm::vec3 location,
          glm::vec3 color,
          glm::vec2 brightness) : location(location),
        color(color),
        brightness(brightness) {}
    glm::vec3 location;
    glm::vec3 color;
    glm::vec2 brightness;
};

class Intersectable;

struct Intersection {
    __host__ __device__ Intersection() : intersected(false), t(INFINITY) {}
    bool intersected;
    glm::vec3 point;
    float t;
    glm::vec3 normal;
    Intersectable *obj;
    Material mat;
};

class DeviceIntersectable {
public:
    virtual __device__ glm::vec3 getNormal(Ray &r, float &t) { return glm::vec3(); }
    virtual __device__ Intersection intersect(Ray &r) {
        return Intersection();
    }
};

class Intersectable {
public:
    virtual Material getMaterial() { return Material(); }
    virtual glm::vec3 getNormal(Ray &r, float &t) { return glm::vec3(); }
    virtual Intersection intersect(Ray &r) {
        return Intersection();
    }
    virtual DeviceIntersectable* returnDeviceIntersectable() {
        return new DeviceIntersectable;
    }
};

class Scene;

struct Work {
    Framebuffer fb;
    size_t yfirst;
    size_t ylast;
};

class OwnedHandle {
    friend class Scene;
private:
    OwnedHandle(std::vector<size_t> ids) : identifiers_owned(ids) {}
    std::vector<size_t> identifiers_owned;
};

class MeshBuilder;

enum ObjectType {
    TriangleType,
    SphereType
};

struct IntersectableInternalOptimized {
    ObjectType type;
    Material mat;
};

struct TriangleInternalOptimized : public IntersectableInternalOptimized{
    glm::vec4 u, v;
    glm::vec4 normal;
};

struct SphereInternalOptimized : public IntersectableInternalOptimized {
    float radius;
    glm::vec3 origin;
};

enum RenderBackend {
    Rendertape,
    Embree,
    OpenCL,
    CUDA,
    Phi
};

extern std::string BackendName[];

class RendererInitializationException : public std::exception {
    virtual const char* what() const throw() {
        return "Renderer failed to initialize";
    }
};

#endif // STRUCTS_H
