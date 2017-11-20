#include <fstream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <sys/time.h>
#include <string>

struct Vec3 {
    float x,y,z;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator + (const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator - (const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator * (float d) const { return Vec3(x*d, y*d, z*d); }
    Vec3 operator / (float d) const { return Vec3(x/d, y/d, z/d); }
    Vec3 normalize() const {
        float mg = sqrtf(x*x + y*y + z*z);
        return Vec3(x/mg,y/mg,z/mg);
    }
};
inline float dot(const Vec3& a, const Vec3& b) {
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}

struct Ray {
    Vec3 o,d;
    Ray(const Vec3& o, const Vec3& d) : o(o), d(d) {}
};

struct Sphere {
    Vec3 c;
    float r;
    Sphere(const Vec3& c, float r) : c(c), r(r) {}
    Vec3 getNormal(const Vec3& pi) const { return (pi - c) / r; }
    bool intersect(const Ray& ray, float &t) const {
        const Vec3 o = ray.o;
        const Vec3 d = ray.d;
        const Vec3 oc = o - c;
        const float b = 2 * dot(oc, d);
        const float c = dot(oc, oc) - r*r;
        float disc = b*b - 4 * c;
        if (disc < 1e-4) return false;
        disc = sqrtf(disc);
        const float t0 = (-b - disc) / 2;
        const float t1 = (-b + disc) / 2;
        t = (t0 < t1) ? t0 : t1;
        return true;
    }
};

void clamp255(Vec3& col) {
    col.x = (col.x > 255) ? 255 : (col.x < 0) ? 0 : col.x;
    col.y = (col.y > 255) ? 255 : (col.y < 0) ? 0 : col.y;
    col.z = (col.z > 255) ? 255 : (col.z < 0) ? 0 : col.z;
}

int main() {
    glewInit();
    glfwInit();
    GLFWwindow *window;
    window = glfwCreateWindow(1920, 1080, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    const int H = 1920;
    const int W = 1080;

    const Vec3 white(255, 255, 255);
    const Vec3 black(0, 0, 0);
    const Vec3 red(255, 0, 0);
    const Vec3 blue(0, 0, 255);

    Sphere sphere(Vec3(W*0.5, H*0.5, 50), 50);
    Sphere sphere2(Vec3(W*0.5, H*0.5, 10), 20);
    const Sphere light(Vec3(0, 0, 50), 1);
    float t;
    Vec3 pix_col(black);
    uint8_t fb[1920*1080*3];
    timeval past, present;
    gettimeofday(&past, NULL);
    int frame = 0;
    int prevframe = frame;
    while(!glfwWindowShouldClose(window)) {
        frame++;
        for (int y = 0; y < H; ++y) {
            #pragma omp for ordered schedule(dynamic)
            for (int x = 0; x < W; ++x) {
                pix_col = black;

                const Ray ray(Vec3(x,y,0),Vec3(0,0,1));
                bool s1;
                if (s1 = sphere.intersect(ray, t)) {
                    const Vec3 pi = ray.o + ray.d*t;
                    const Vec3 L = light.c - pi;
                    const Vec3 N = sphere.getNormal(pi);
                    const float dt = dot(L.normalize(), N.normalize());

                    pix_col = (red + white*dt) * 0.5;
                    clamp255(pix_col);
                }
                float told = t;
                if(sphere2.intersect(ray, t)){
                    const Vec3 pi = ray.o + ray.d*t;
                    const Vec3 L = light.c - pi;
                    const Vec3 N = sphere2.getNormal(pi);
                    const float dt = dot(L.normalize(), N.normalize());
                    if(!s1) {
                        pix_col = (blue + white*dt) * 0.5;
                        clamp255(pix_col);
                    } else if(t < told){
                        pix_col = ((pix_col) + ((blue + white*dt) * 0.5))/2;
                        clamp255(pix_col);
                    }
                }
                #pragma omp ordered
                fb[(x*(W*3)) + (y*3)] = (int)pix_col.x;
                fb[(x*(W*3)) + (y*3) + 1] = (int)pix_col.y;
                fb[(x*(W*3)) + (y*3) + 2] = (int)pix_col.z;
            }
        }
        glDrawPixels(W, H, GL_RGB, GL_UNSIGNED_BYTE, fb);
        glfwSwapBuffers(window);
        glfwPollEvents();
        glfwSwapInterval(1);
        gettimeofday(&present, NULL);
        if(present.tv_sec > past.tv_sec) {
            int fps = (frame - prevframe);
            std::string title = "t - " + std::to_string(fps) + " FPS";
            prevframe = frame;
            glfwSetWindowTitle(window, title.c_str());
            past = present;
        }
        sphere2 = Sphere(Vec3(W*0.5, (H*0.5)+cos(frame/16.0)*100, 50+sin(frame/16.0)*100), 20);
    }
}
