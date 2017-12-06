#include <glm/glm.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <sys/time.h>
#include <thread>
#include <semaphore.h>
#include <atomic>
#include <algorithm>
#include <future>
#include <functional>
#include <unordered_map>
#include "ctpl.h"
#include <time.h>
#include <set>
#include <cstdlib>
#include <cuda/cuda.h>
#include <cuda/cuda_runtime.h>
#include <cuda/thrust/device_vector.h>
#ifndef __global__
 #define __global__
#endif

class SceneManager;

void printVec3(glm::vec3 _v1) {
    std::cout << "(" << _v1.x << ", " << _v1.y << ", " << _v1.z << ")" << std::endl;
}

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
    size_t bounces;
};

struct Framebuffer {
    size_t x;
    size_t y;
    uint8_t *fb;
};

struct Material {
    glm::vec3 color;
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
    __host__ __device__ Intersection() : intersected(false), t(0.0f) {}
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
    virtual glm::vec3 getNormal(Ray &r, float &t) { return glm::vec3(); }
    virtual Intersection intersect(Ray &r) {
        return Intersection();
    }
    virtual DeviceIntersectable* returnDeviceIntersectable() {
        return new DeviceIntersectable;
    }
};
class DeviceTriangle : public DeviceIntersectable {
    friend class MeshBuilder;
public:
    __host__ __device__ Triangle(glm::vec3 pts[3], glm::vec3 color) : color(color) {
        memcpy(this->pts, pts, sizeof(glm::vec3)*3);
        glm::vec3 u = pts[1] - pts[0];
        glm::vec3 v = pts[2] - pts[0];
        for(size_t i = 0; i < 3; i++) {
            normals[i] = glm::normalize(glm::cross(u, v));
        }
    }

    __device__ void setNormal(glm::vec3 nls[3]) {
        memcpy(normals, nls, sizeof(glm::vec3)*3);
    }

    __device__ virtual glm::vec3 getNormal(Ray &r, float &t) {
        glm::vec3 pt = r.origin + r.direction*t;
        glm::vec3 nearest;
        float dist = INFINITY;
        for(glm::vec3 n : normals) {
            float cdist = glm::distance(pt, n);
            if(cdist < dist) {
                nearest = n;
                dist = cdist;
            }
        }
        return nearest;
    }

    __device__ virtual Intersection intersect(Ray &r) {
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
        ret.normal = getNormal(r, ret.t);
        Material mat;
        mat.color = color;
        ret.mat = mat;
        ret.obj = this;
        return ret;
    }

private:
    glm::vec3 pts[3];
    glm::vec3 normals[3];
    glm::vec3 color;
};

struct Work {
    Framebuffer fb;
    size_t yfirst;
    size_t ylast;
};

class OwnedHandle {
    friend class SceneManager;
private:
    OwnedHandle(std::vector<size_t> ids) : identifiers_owned(ids) {}
    std::vector<size_t> identifiers_owned;
};

class MeshBuilder;

class Triangle : public Intersectable {
    friend class MeshBuilder;
public:
    Triangle(glm::vec3 pts[3], glm::vec3 color) : color(color) {
        memcpy(this->pts, pts, sizeof(glm::vec3)*3);
        glm::vec3 u = pts[1] - pts[0];
        glm::vec3 v = pts[2] - pts[0];
        for(size_t i = 0; i < 3; i++) {
            normals[i] = glm::normalize(glm::cross(u, v));
        }
    }

    void setNormal(glm::vec3 nls[3]) {
        memcpy(normals, nls, sizeof(glm::vec3)*3);
    }

    virtual glm::vec3 getNormal(Ray &r, float &t) {
        glm::vec3 pt = r.origin + r.direction*t;
        glm::vec3 nearest;
        float dist = INFINITY;
        for(glm::vec3 n : normals) {
            float cdist = glm::distance(pt, n);
            if(cdist < dist) {
                nearest = n;
                dist = cdist;
            }
        }
        return nearest;
    }

    virtual Intersection intersect(Ray &r) {
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
        ret.normal = getNormal(r, ret.t);
        Material mat;
        mat.color = color;
        ret.mat = mat;
        ret.obj = this;
        return ret;
    }

    virtual DeviceIntersectable* returnDeviceIntersectable() {
        return new DeviceTriangle(pts, color);
    }

private:
    glm::vec3 pts[3];
    glm::vec3 normals[3];
    glm::vec3 color;
};

struct CmpClass // class comparing vertices in the set
{
    bool operator() (const std::pair<glm::vec3, int>& p1, const std::pair<glm::vec3, int>& p2) const
    {
        if (fabs(p1.first.x-p2.first.x) > eps) return p1.first.x < p2.first.x;
        if (fabs(p1.first.y-p2.first.y) > eps) return p1.first.y < p2.first.y;
        if (fabs(p1.first.z-p2.first.z) > eps) return p1.first.z < p2.first.z;
        return false;
    }
    float eps = 0.0001;
};

class MeshBuilder {
public:
    MeshBuilder(std::vector<Triangle> tris) : tris(tris) {}

    void GenerateNormals() {
        std::vector<glm::vec3> vertices;
        std::vector<int> indices;
        for(Triangle tri : tris) {
            vertices.push_back(tri.pts[0]);
            vertices.push_back(tri.pts[1]);
            vertices.push_back(tri.pts[2]);
        }
        std::set<std::pair<glm::vec3, int>, CmpClass> vpairs;
        int index =  0;
        for(size_t i = 0; i < vertices.size(); i++) {
            std::set<std::pair<glm::vec3, int>>::iterator it = vpairs.find(std::make_pair(vertices[i], 0));
            if(it != vpairs.end()) {
                indices.push_back(it->second);
            } else {
                vpairs.insert(std::make_pair(vertices[i], index));
                indices.push_back(index++);
            }
        }
        std::vector<glm::vec3> nverts;
        nverts.resize(vpairs.size());
        for (auto &kv : vpairs)
               nverts[kv.second] = kv.first;
        std::vector<glm::vec3> normals;
        normals.resize(nverts.size());
        for(size_t i = 0; i < indices.size(); i += 3) {
            glm::vec3 e1 = nverts[indices[i+1]] - nverts[indices[i]];
            glm::vec3 e2 = nverts[indices[i+2]] - nverts[indices[i]];
            glm::vec3 no = glm::cross(e1, e2);
            normals[indices[i]] += no;
            normals[indices[i+1]] += no;
            normals[indices[i+2]] += no;
        }
        for(glm::vec3 &n : normals)
            n = glm::normalize(n);
        std::vector<Triangle> newtris;
        for(size_t i = 0; i < indices.size(); i+= 3) {
            std::vector<glm::vec3> points;
            std::vector<glm::vec3> ntri;
            for(size_t j = 0; j < 3; j++) {
                points.push_back(nverts[indices[i+j]]);
                ntri.push_back(normals[indices[i+j]]);
            }
            Triangle t(points.data(), glm::vec3(1.0, 1.0, 1.0));
            t.setNormal(ntri.data() + (i*3));
            newtris.push_back(t);
        }
        tris = newtris;

    }

    std::vector<Triangle> GetTriangles() {
        return tris;
    }

private:
    std::vector<Triangle> tris;
};

class Sphere : public DeviceIntersectable {
public:
    __host__ Sphere(glm::vec3 origin, float radius, glm::vec3 color) : origin(origin), radius(radius), color(color) {}
    __device__ glm::vec3 getNormal(Ray &ray, float &t) { return ((ray.origin + (t*ray.direction) - origin)) / radius; }
    __device__ Intersection intersect(Ray &ray) {
        Intersection ret;
        glm::vec3 p = ray.origin - origin;
        float rpow2 = radius*radius;
        float p_d = glm::dot(p, ray.direction);

        if(p_d > 0 || dot(p, p) < rpow2)
            return ret;

        glm::vec3 a = p - p_d * ray.direction;
        float apow2 = glm::dot(a, a);
        if(apow2 > rpow2)
            return ret;
        ret.intersected = true;
        float h = sqrtf(rpow2 - apow2);
        glm::vec3 i = a - h*ray.direction;
        ret.point = origin + i;
        ret.t = ((ret.point - ray.origin) / ray.direction).x;
        ret.normal = i/radius;
        Material mat;
        mat.color = color;
        ret.mat = mat;
        return ret;

    }
private:
    glm::vec3 origin;
    float radius;
    glm::vec3 color;
};

class Sphere : public Intersectable {
public:
    Sphere(glm::vec3 origin, float radius, glm::vec3 color) : origin(origin), radius(radius), color(color) {}
    glm::vec3 getNormal(Ray &ray, float &t) { return ((ray.origin + (t*ray.direction) - origin)) / radius; }
    Intersection intersect(Ray &ray) {
        Intersection ret;
        glm::vec3 p = ray.origin - origin;
        float rpow2 = radius*radius;
        float p_d = glm::dot(p, ray.direction);

        if(p_d > 0 || dot(p, p) < rpow2)
            return ret;

        glm::vec3 a = p - p_d * ray.direction;
        float apow2 = glm::dot(a, a);
        if(apow2 > rpow2)
            return ret;
        ret.intersected = true;
        float h = sqrtf(rpow2 - apow2);
        glm::vec3 i = a - h*ray.direction;
        ret.point = origin + i;
        ret.t = ((ret.point - ray.origin) / ray.direction).x;
        ret.normal = i/radius;
        Material mat;
        mat.color = color;
        ret.mat = mat;
        return ret;

    }
private:
    glm::vec3 origin;
    float radius;
    glm::vec3 color;
};

__global__
void castRay(Ray r, DeviceIntersectable* devptr, size_t nitems) {
    if(threadIdx.x < nitems) {
        DeviceIntersectable *curitem = devptr + threadIdx.x;
        curitem->intersect(r);
    }
}


class SceneManager {
public:
    SceneManager() : pool(8) {
        current_id = 0;
        fast_srand(0);
    }

    OwnedHandle AddObject(Intersectable* obj) {
        intersectables[current_id] = obj;
        OwnedHandle h(std::vector<size_t>(1, current_id));
        current_id++;
        RegenerateObjectCache();
        return h;
    }

    OwnedHandle AddObject(std::vector<Intersectable*> objects) {
        std::vector<size_t> handles;
        for(Intersectable* obj : objects) {
            handles.push_back(current_id);
            intersectables[current_id] = obj;
            current_id++;
        }
        RegenerateObjectCache();
        return OwnedHandle(handles);
    }

    void RemoveObjectsByHandle(OwnedHandle handle) {
        for(size_t i : handle.identifiers_owned) {
            intersectables.erase(i);
        }
        RegenerateObjectCache();
    }

    void RegenerateObjectCache() {
        intersectables_cached.clear();
        for(auto kv : intersectables) {
            Intersectable *obj = kv.second;
            intersectables_cached.push_back(obj);
        }
    }

    void AddLight(Light* obj) {
        lights.push_back(obj);
    }
    
    void SetCameraConfig(CameraConfig &cfg) {
        config = cfg;
    }

    glm::vec3 GenerateNoiseVector() {
        int x = fast_rand() % UINT16_MAX;
        int y = fast_rand() % UINT16_MAX;
        int z = fast_rand() % UINT16_MAX;
        return glm::vec3(x, y, z);
    }

    void cast(Ray &r, std::vector<Intersection> *hits, bool &anyintersection) {
        for(auto obj : intersectables_cached) {
            Intersection hit = obj->intersect(r);
            if(hit.intersected) {
                hits->push_back(hit);
                anyintersection = true;
            }
        }
    }

    void RenderSlice(size_t yfirst, size_t ylast, Framebuffer &fb) {
        glm::vec3 camright = glm::cross(config.up,config.lookat);
        glm::vec3 localup = glm::cross(camright, config.lookat);
        std::vector<Intersection> hits;
        std::vector<Intersection> shadow_hits;
        float aspect = fb.x/(float)fb.y;
        glm::vec3 correctedright = aspect * camright;
        hits.reserve(100);
        shadow_hits.reserve(100);
        for(size_t x = 0; x < fb.x; x++) {
            for(size_t y = yfirst; y < ylast; y++) {
                float nx = ((float)x / fb.x) - 0.5;
                float ny = ((float)y / fb.y) - 0.5;
                Ray r;
                r.origin = config.center;
                r.direction = (correctedright * nx) + (-localup * ny) + config.lookat;
                r.direction = glm::normalize(r.direction);
                bool anyintersection = false;
                cast(r, &hits, anyintersection);
                if(!anyintersection) {
                    fb.fb[((fb.x * y) + x)*3] = 135;
                    fb.fb[((fb.x * y) + x)*3 + 1] = 206;
                    fb.fb[((fb.x * y) + x)*3 + 2] = 235;
                    continue;
                }
                std::sort(hits.begin(), hits.end(), [](Intersection a, Intersection b){
                    return b.t > a.t;
                });
                glm::vec3 fcolor;
                bool lit = false;
                for(Light *light : lights) {
                    glm::vec3 l = light->location - hits[0].point;
                    glm::vec3 n = hits[0].normal;
                    if(glm::dot(n, -r.direction) < 0) {
                        n = -n;
                    }
                    float dt = glm::dot(glm::normalize(l), n);
                    float albedo = 0.18 / 3.14f;
                    Ray s;
                    s.origin = glm::vec3(hits[0].point) + (n * 0.001f);
                    glm::vec3 noise = GenerateNoiseVector();
                    s.direction = glm::normalize((light->location - hits[0].point) + noise/(float)(UINT16_MAX*2));
                    bool r = false;
                    //cast(s, &shadow_hits, r);
                    std::sort(shadow_hits.begin(), shadow_hits.end(), [](Intersection a, Intersection b){
                        return b.t > a.t;
                    });
                    if((!r) || (glm::distance(shadow_hits[0].point, s.origin) > glm::distance(s.origin, light->location))) {
                        fcolor += (((light->color*dt))* hits[0].mat.color);
                        lit = true;
                    }
                    shadow_hits.clear();
                }

                //fcolor.x = pow(fcolor.x, 1.0 / 2.2);
                //fcolor.y = pow(fcolor.y, 1.0 / 2.2);
                //fcolor.z = pow(fcolor.z, 1.0 / 2.2);

                if (!lit) {
                    std::memset(fb.fb + ((fb.x * y) + x)*3, 0 , 3);
                    hits.clear();
                    continue;
                }
                //fcolor /= lights.size() + 1;
                fb.fb[((fb.x * y) + x)*3] = fminf(fmaxf(0,fcolor.x*255),255);
                fb.fb[((fb.x * y) + x)*3 + 1] = fminf(fmaxf(0,fcolor.y*255),255);
                fb.fb[((fb.x * y) + x)*3 + 2] = fminf(fmaxf(0,fcolor.z*255),255);
                hits.clear();

            }
        }
    }
    
    void render(Framebuffer &fb) {
        if(!(fb.x) || !(fb.y)) {
            return;
        }
        size_t ystep = fb.y/8;
        std::vector<std::future<void>> f(8);
        for(size_t i = 0; i < 8; i++) {
            f[i] = pool.push(std::bind(&SceneManager::RenderSlice, this, ystep*i, ystep*(i+1), fb));
        }
        for(size_t i = 0; i < 8; i++) {
            f[i].get();
        }
        //RenderSlice(0, fb.y, fb);
        //RenderSlice(0, fb.y/2, fb);
        //RenderSlice(fb.y/2, fb.y, fb);
    }

    inline void fast_srand(int seed) {
        g_seed = seed;
    }

    // Compute a pseudorandom integer.
    // Output value in range [0, 32767]
    inline int fast_rand(void) {
        g_seed = (214013*g_seed+2531011);
        return (g_seed>>16)&0x7FFF;
    }
private:
    unsigned int g_seed;
    bool frun = true;
    std::atomic<size_t> current_id;
    CameraConfig config;
    std::unordered_map<size_t, Intersectable*> intersectables;
    std::vector<Intersectable*> intersectables_cached;
    std::vector<Light*> lights;
    ctpl::thread_pool pool;

};

int main() {
    glewInit();
    glfwInit();
    GLFWwindow *window;
    window = glfwCreateWindow(1280, 720, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    glm::vec3 tris[3] = {glm::vec3(40.0, -3.0, -40.0), glm::vec3(-40.0, -3.0, 40.0), glm::vec3(-40.0, -3.0, -40.0)};
    Triangle *tri = new Triangle(tris, glm::vec3(1.0, 0.0, 1.0));
    glm::vec3 tris2[3] = {glm::vec3(40.0, -3.0, -40.0), glm::vec3(-40.0, -3.0, 40.0), glm::vec3(40.0, -3.0, 40.0)};
    Triangle *tri2 = new Triangle(tris2, glm::vec3(1.0, 0.0, 1.0));
    Sphere *s = new Sphere(glm::vec3(0.0, 0.0, 0.0), 2, glm::vec3(0.0, 0.0, 1.0));
    Sphere *s2 = new Sphere(glm::vec3(10.0, 0.0, 0.0), 2, glm::vec3(0.0, 1.0, 0.0));
    Sphere *s3 = new Sphere(glm::vec3(10.0, 0.0, 10.0), 2, glm::vec3(1.0, 0.0, 0.0));
    Light *l = new Light(glm::vec3(-10.0, 8.0, -10.0), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 0.20));
    Light *l2 = new Light(glm::vec3(10, 10, 10), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 0.20));

    std::vector<Triangle> triangles;
    triangles.push_back(*tri);
    triangles.push_back(*tri2);
    //MeshBuilder b(triangles);
    //b.GenerateNormals();
    //Sphere *s2 = new Sphere(glm::vec3(0.0, -4.0, 0.0), 4);
    SceneManager man;
    man.AddObject(s);
    man.AddObject(s2);
    man.AddObject(s3);
    man.AddObject(tri);
    man.AddObject(tri2);
    man.AddLight(l);
    man.AddLight(l2);
    CameraConfig cfg;
    cfg.center = glm::vec3(16.0, 4.0, 16.0);
    printVec3(cfg.center);
    cfg.lookat  = glm::normalize(glm::vec3(0.0, 0.0, 0.0) - cfg.center);
    printVec3(cfg.lookat);
    cfg.up   = glm::vec3(1.0, 0.0, 0.0);
    printVec3(cfg.up);
    man.SetCameraConfig(cfg);
    Framebuffer fb;
    fb.x = 1280;
    fb.y = 720;
    fb.fb = (uint8_t*)malloc(fb.x*fb.y*3);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLfloat) 100, 0.0, (GLfloat) 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    timeval past, present;
    gettimeofday(&past, NULL);
    glfwSetCursorPos(window, fb.x/2, fb.y/2);
    glfwSetInputMode(window, GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
    glPixelStorei(GL_PACK_ALIGNMENT, 8);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    int frame = 0;
    int prevframe = frame;
    float horizontal = 3.14f;
    float vertical = 0.0f;
    float mspeed = 0.005f;
    cudaDeviceSynchronize();
    while(!glfwWindowShouldClose(window)) {
        l->location.x = sin(frame/32.0)*20;
        l->location.z = cos(frame/32.0)*20;
        double xpos = 0, ypos = 0;
        glfwGetCursorPos(window, &xpos, &ypos);
        glfwSetCursorPos(window, fb.x/2, fb.y/2);
        horizontal += mspeed * -(fb.x/2 - xpos);
        vertical += mspeed * (fb.y/2 - ypos);

        if (vertical > 1.5f) {
            vertical = 1.5f;
        }
        else if (vertical < -1.5f) {
            vertical = -1.5f;
        }
        cfg.lookat = glm::vec3(cos(vertical) * sin(horizontal), sin(vertical), cos(horizontal) * cos(vertical));
        glm::vec3 right = glm::vec3(sin(horizontal - 3.14f / 2.0f), 0, cos(horizontal - 3.14f / 2.0f));
        cfg.up = glm::cross(right, cfg.lookat);
        if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cfg.center += cfg.lookat;
        }
        if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cfg.center -= cfg.lookat;
        }
        if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cfg.center += right;
        }
        if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cfg.center -= right;
        }
        cudaGetLastError();
        man.SetCameraConfig(cfg);
        man.render(fb);
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(0,0);
        glDrawPixels(fb.x, fb.y, GL_RGB, GL_UNSIGNED_BYTE, fb.fb);
        glfwSwapBuffers(window);
        glfwPollEvents();
        gettimeofday(&present, NULL);
        if(present.tv_sec > past.tv_sec) {
            int fps = (frame - prevframe);
            std::string title = "t - " + std::to_string(fps) + " FPS";
            prevframe = frame;
            glfwSetWindowTitle(window, title.c_str());
            past = present;
        }
        frame++;
    }
}
