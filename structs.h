#ifndef STRUCTS_H
#define STRUCTS_H
#include <glm/glm.hpp>
#include "cudapatch.h"
#include <vector>
#include <string>
#include <CL/cl.h>
#include <GL/glew.h>

enum MatType {
    DIFFUSE_GLOSS = 0,
    REFLECT_REFRACT = 1,
    REFLECT = 2
};

enum AreaLightType {
    TRIANGLELIST,
    SPHERELIST
};

struct AreaLightCL {
    cl_uint emitliststart;
    cl_uint emitters;
    AreaLightType type;
};


struct CameraConfigCL {
    cl_float3 center;
    cl_float3 lookat;
    cl_float3 up;
};

struct MaterialCL {
    cl_float3 color;
    MatType type;
    float rindex;
    float diffc;
    float specc;
    float specexp;
    float emits;
};

struct LightCL {
    cl_float3 pos;
    cl_float3 color;
};

struct SphereCL {
    MaterialCL mat;
    cl_float3 origin;
    cl_float radius;
};

struct TriangleCL {
    MaterialCL mat;
    cl_float3 pts[3];
};

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
    GLuint textureid;
};

struct Material {
    glm::vec3 color;
    MatType type;
    float rindex;
    float diffc;
    float specc;
    float specexp;
    float emits = 0.0;
    int alightid;
};

struct Light {
    Light(glm::vec3 origin, glm::vec3 color,
          glm::vec2 brightness) : color(color), brightness(brightness) {
        transform[3][0] = origin.x;
        transform[3][1] = origin.y;
        transform[3][2] = origin.z;
    }
    glm::mat4x4 transform;
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

class Intersectable {
public:
    virtual Material getMaterial() { return Material(); }
    virtual glm::vec3 getNormal(Ray &r, float &t) { return glm::vec3(); }
    virtual Intersection intersect(Ray &r) { return Intersection(); }
    virtual void setTransform(glm::mat4x4 m) { transform = m; }
    virtual glm::mat4x4 getTransform() const { return transform; }
protected:
    glm::mat4x4 transform = glm::mat4x4(1.0f);
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
public:
    RendererInitializationException(std::string error) : error(error) {}
    virtual const char* what() const throw() {
        std::string err = "Renderer failed to compile OpenCL code: \n";
        err += error;
        return error.c_str();
    }
private:
    std::string error;
};

#endif // STRUCTS_H
