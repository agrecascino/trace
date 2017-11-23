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

glm::vec3 crossVec4(glm::vec3 _v1, glm::vec3 _v2){
    glm::vec3 vec1 = glm::vec3(_v1[0], _v1[1], _v1[2]);
    glm::vec3 vec2 = glm::vec3(_v2[0], _v2[1], _v2[2]);
    glm::vec3 res = glm::cross(vec1, vec2);
    return glm::vec3(res[0], res[1], res[2]);
}


void printVec4(glm::vec3 _v1) {
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

class Material {

};

class Intersectable;

struct Intersection {
    Intersection() : intersected(false), t(0.0f) {}
    bool intersected;
    glm::vec3 point;
    float t;
    glm::vec3 normal;
    Intersectable *obj;
};

class Intersectable {
public:
    virtual glm::vec3 getNormal(Ray &r, float &t) { return glm::vec3(); }
    virtual Intersection intersect(Ray &r) {
        return Intersection();
    }
};

struct Work {
    size_t nthreads;
    Framebuffer fb;
    sem_t sem;
    bool stop = false;
};

class Sphere : public Intersectable {
public:
    Sphere(glm::vec3 origin, float radius) : origin(origin), radius(radius) {}
    glm::vec3 getNormal(Ray &ray, float &t) { return ((ray.origin + (t*ray.direction) - origin)) / radius; }
    Intersection intersect(Ray &ray) {
        Intersection ret;
        ret.intersected = false;
        glm::vec3 m = ray.origin - origin;
        float b = glm::dot(m, ray.direction);
        float c = glm::dot(m, m) - radius * radius;

        // Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0)
        if (c > 0.0f && b > 0.0f) return ret;
        float discr = b*b - c;

        // A negative discriminant corresponds to ray missing sphere
        if (discr < 0.0f) return ret;

        // Ray now found to intersect sphere, compute smallest t value of intersection
        ret.t = -b - sqrtf(discr);
        ret.intersected = true;

        // If t is negative, ray started inside sphere so clamp t to zero
        if (ret.t < 0.0f) ret.t = 0.0f;
        ret.point = ray.origin + ret.t * ray.direction;
        ret.normal = getNormal(ray, ret.t);
        return ret;

    }
private:
    glm::vec3 origin;
    float radius;
};

class SceneManager {
public:
    void AddObject(Intersectable* obj) {
        intersectables.push_back(obj);
    }
    
    void SetCameraConfig(CameraConfig cfg) {
        config = cfg;
    }

    void RenderWorker(int thread) {
        while(!work[thread].stop) {
            sem_wait(&work[thread].sem);
            size_t xstep = work[thread].nthreads;
            size_t xfirst = xstep * thread;
            size_t xlast = ((xstep+1) * thread);
            glm::vec3 camright = crossVec4(config.up,config.lookat);
            config.up = crossVec4(camright, config.lookat);
            //std::vector<Intersection> hits;
            //hits.reserve(100);
            Framebuffer fb = work[thread].fb;
            for(size_t x = xfirst; x < xlast; x++) {
                for(size_t y = 0; y < fb.y; y++) {
                    float nx = ((float)x / fb.x) - 0.5;
                    float ny = ((float)y / fb.y) - 0.5;
                    Ray r;
                    r.origin = config.center;
                    r.direction = (camright*((float)fb.x/fb.y) * nx) + (-config.up * ny) + config.center + config.lookat;
                    r.direction = glm::normalize(r.direction - r.origin);
                    /*std::cout << "Ray shot from (" << r.origin.x << ", " << r.origin.y << ", "
                                  << r.origin.z << ")" << " direction was (" << r.direction.x << ", "
                                  << r.direction.y << ", " << r.direction.z << ")" << std::endl;*/
                    for(Intersectable *obj : intersectables) {
                        Intersection hit = obj->intersect(r);
                        if(hit.intersected) {
                            /*std::cout << "Hit at point (" << hit.point.x << ", " << hit.point.y
                                                              << ", " << hit.point.z << ") t = "
                                                              << hit.t << std::endl;*/
                            //hits.push_back(hit);
                            fb.fb[((fb.x * y) + x)*3] = hit.normal.x*255;
                            fb.fb[((fb.x * y) + x)*3 + 1] = hit.normal.y*255;
                            fb.fb[((fb.x * y) + x)*3 + 2] = hit.normal.z*255;
                        } else {
                            std::memset(fb.fb + (y*fb.x*3) + (x*3), 0, 3);
                        }
                    }
                    //hits.clear();
                }
            }
        }
    }
    
    void render(Framebuffer &fb) {
        if(!(fb.x) || !(fb.y)) {
            return;
        }
        if(firstrun) {
            for(size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
                Work wk;
                sem_init(&wk.sem, 0, 0);
                wk.nthreads = std::thread::hardware_concurrency();
                work.push_back(wk);
                workers.push_back(std::thread(std::bind(&SceneManager::RenderWorker, this, i)));
                workers[i].detach();
            }
            firstrun = false;
        }
        for(size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
            work[i].fb = fb;
            sem_post(&work[i].sem);
        }
        for(size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
            while(true) {
                int s;
                if(sem_getvalue(&work[i].sem, &s)) {
                    continue;
                }
                if(!s) {
                    break;
                }
            }
        }
    }
private:
    bool firstrun = true;
    std::vector<Work> work;
    std::vector<std::thread> workers;
    CameraConfig config;
    std::vector<Intersectable*> intersectables;
};

int main() {
    glewInit();
    glfwInit();
    GLFWwindow *window;
    window = glfwCreateWindow(1920, 1080, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    Sphere *s = new Sphere(glm::vec3(0.0, 0.0, 0.0), 2);
    //Sphere *s2 = new Sphere(glm::vec3(0.0, -4.0, 0.0), 4);
    SceneManager man;
    man.AddObject(s);
    //man.AddObject(s2);
    CameraConfig cfg;
    cfg.center = glm::vec3(16.0, 4.0, 16.0);
    printVec4(cfg.center);
    cfg.lookat  = glm::normalize(glm::vec3(0.0, 0.0, 0.0) - cfg.center);
    printVec4(cfg.lookat);
    cfg.up   = glm::vec3(1.0, 0.0, 0.0);
    printVec4(cfg.up);
    man.SetCameraConfig(cfg);
    Framebuffer fb;
    fb.x = 1920;
    fb.y = 1080;
    fb.fb = (uint8_t*)malloc(fb.x*fb.y*3);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLfloat) 100, 0.0, (GLfloat) 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    timeval past, present;
    gettimeofday(&past, NULL);
    int frame = 0;
    int prevframe = frame;
    while(!glfwWindowShouldClose(window)) {
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
