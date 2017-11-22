#include <glm/glm.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

glm::vec4 crossVec4(glm::vec4 _v1, glm::vec4 _v2){
    glm::vec3 vec1 = glm::vec3(_v1[0], _v1[1], _v1[2]);
    glm::vec3 vec2 = glm::vec3(_v2[0], _v2[1], _v2[2]);
    glm::vec3 res = glm::cross(vec1, vec2);
    return glm::vec4(res[0], res[1], res[2], 1);
}

void printVec4(glm::vec4 _v1) {
    std::cout << "(" << _v1.x << ", " << _v1.y << ", " << _v1.z << ")" << std::endl;
}

struct CameraConfig {
    CameraConfig() {}
    CameraConfig(glm::vec4 center, glm::vec4 lookat,
                 glm::vec4 up) : center(center),
                 lookat(lookat), up(up) {}
                
    glm::vec4 center;
    glm::vec4 lookat;
    glm::vec4 up;
};

struct Ray {
    glm::vec4 origin;
    glm::vec4 direction;
    size_t bounces;
};

struct Framebuffer {
    size_t x;
    size_t y;
    uint8_t *fb;
};

class Intersectable;

struct Intersection {
    Intersection() : intersected(false), t(0.0f) {}
    bool intersected;
    glm::vec4 point;
    float t;
    glm::vec4 normal;
    Intersectable *obj;
};

class Intersectable {
public:
    virtual glm::vec4 getNormal(Ray &r, float &t) { return glm::vec4(); }
    virtual Intersection intersect(Ray &r) {
        return Intersection();
    }
};


class Sphere : public Intersectable {
public:
    Sphere(glm::vec4 origin, float radius) : origin(origin), radius(radius) {}
    glm::vec4 getNormal(Ray &ray, float &t) { return ((ray.origin + (t*ray.direction) - origin)) / radius; }
    Intersection intersect(Ray &ray) {
        Intersection ret;
        glm::vec4 m = ray.origin - origin;
        float b = 2*glm::dot(m, ray.direction);
        float c = glm::dot(m, m) - (radius*radius);
        if((c > 0.0f) && (b > 0.0f)) return ret;
        float discr = (b*b) - (4*c);
        if(discr < 1e-4) return ret;
        float t = (-b - sqrtf(discr)) / 2;
        if(t < 0.0f) t = 0.0f;
        ret.point = ray.origin + (t * ray.direction);
        ret.t = t;
        ret.intersected = true;
        ret.normal = getNormal(ray, t);
        ret.obj = this;
        return ret;
    }
private:
    float radius;
    glm::vec4 origin;
};

class SceneManager {
public:
    void AddObject(Intersectable* obj) {
        intersectables.push_back(obj);
    }
    
    void SetCameraConfig(CameraConfig cfg) {
        config = cfg;
    }
    
    void render(Framebuffer &fb) {
        if(!(fb.x) || !(fb.y)) {
            return;
        }
        fb.fb = (uint8_t*)malloc(fb.x*fb.y*3);
        glm::vec4 camright = crossVec4(config.lookat, config.up); 
        config.up = crossVec4(camright, config.lookat);
        for(size_t x = 0; x < fb.x; x++) {
            for(size_t y = 0; y < fb.y; y++) {
                float nx = ((float)x / fb.x) - 0.5;
                float ny = ((float)y / fb.y) - 0.5;
                Ray r;
                r.origin = config.center;
                glm::vec4 ip = nx * camright + ny * config.up + config.center + config.lookat;
                r.direction = ip - config.center;
                /*std::cout << "Ray shot from (" << r.origin.x << ", " << r.origin.y << ", "
                          << r.origin.z << ")" << " direction was (" << r.direction.x << ", "
                          << r.direction.y << ", " << r.direction.z << ")" << std::endl;  */
                std::vector<Intersection> hits;
                for(Intersectable *obj : intersectables) {
                    Intersection hit = obj->intersect(r);
                    //if(hit.intersected) {
                        //hits.push_back(hit);
                        fb.fb[(y*fb.x) + (x*3)] = 255;
                        fb.fb[(y*fb.x) + (x*3) + 1] = 255;                     
                        fb.fb[(y*fb.x) + (x*3) + 2] = 255;
                    //}
                }
            }
        }
    }
private:
    CameraConfig config;
    std::vector<Intersectable*> intersectables;
};

int main() {
    glewInit();
    glfwInit();
    GLFWwindow *window;
    window = glfwCreateWindow(1920, 1080, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    Sphere *s = new Sphere(glm::vec4(0.0, 0.0, 0.0, 1.0), 1);
    SceneManager man;
    man.AddObject(s);
    CameraConfig cfg;
    cfg.center = glm::vec4(0.0, 14.0, 0.0, 1.0);
    printVec4(cfg.center);
    cfg.lookat  = glm::vec4(0.0, -1.0, 0.0, 1.0);
    printVec4(cfg.lookat);
    cfg.up   = glm::vec4(1.0, 0.0, 0.0, 1.0);
    printVec4(cfg.up);
    man.SetCameraConfig(cfg);
    Framebuffer fb;
    fb.x = 1920;
    fb.y = 1080;
    while(!glfwWindowShouldClose(window)) {
        man.render(fb);
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(0,0);
        glDrawPixels(fb.x, fb.y, GL_RGB, GL_UNSIGNED_BYTE, fb.fb);
        glfwSwapBuffers(window);
        glfwPollEvents();
        glfwSwapInterval(1);
    }
}
