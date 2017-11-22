#include <fstream>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <sys/time.h>
#include <string>
#include <cstring>
#include <

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
        const float docd = dot(oc, d);
        const float b = 2 * docd;
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

struct Triangle {
    Vec3 points[3];
    Vec3 normal;
    Vec3 loc;
    Triangle(const Vec3 points[3], const Vec3 loc) : points(points), loc(loc) { 
        const Vec3 u = points[1] - points[0];
        const Vec3 v = points[2] - points[1];
        Vec3 n;
        n.x = (u.y * v.z) - (u.z - v.y);
        n.y = (u.z * v.x) - (u.x * v.z);
        n.z = (u.x * v.y) - (u.y * v.x);
        normal = n;
    }
    Vec3 getNormal() const { 
        return normal;
    }
    bool intersect(const Ray& ray, float &t) const {
        const Vec3 o = ray.o;
        const Vec3 d = ray.d;
        t = (-dot(points[0], normal + o) ) / (dot(d, normal));
        Vec3 p = o + t*d;
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
    window = glfwCreateWindow(1600, 900, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    const int H = 900;
    const int W = 1600;

    const Vec3 white(255, 255, 255);
    const Vec3 black(0x7e, 0xc0, 0xee);
    const Vec3 red(255, 0, 0);
    const Vec3 blue(0, 0, 255);

    Sphere sphere(Vec3(W*0.5, H*0.5, 50), 50);
    Sphere sphere2(Vec3(W*0.5, H*0.5, 10), 20);
    Sphere sphere3(Vec3(W*0.5, H*0.5, 10), 20);
    Sphere light(Vec3(W*0.5, (H*0.5)+500, 20), 20);
    uint8_t fb[1600*900*3];
    timeval past, present;
    gettimeofday(&past, NULL);
    int frame = 0;
    int prevframe = frame;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLfloat) 100, 0.0, (GLfloat) 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    bool lastfail = true;
    while(!glfwWindowShouldClose(window)) {
        frame++;
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                Vec3 pix_col(black);
                float t = 0;
                pix_col = black;
                const Ray ray(Vec3(x,y,0),Vec3(0,0,1));
                bool s1;
                if (s1 = sphere.intersect(ray, t)) {
                    const Vec3 pi = ray.o + ray.d*t;
                    const Vec3 L = light.c - pi;
                    const Vec3 N = sphere.getNormal(pi);
                    float dt = dot(L.normalize(), N.normalize());
                    dt = fmax(dt,0.01);
                    pix_col = (red + white*dt) * 0.5;
                    clamp255(pix_col);
                }
                bool s2;
                float told = t;
                if(s2 = sphere2.intersect(ray, t)){
                    const Vec3 pi = ray.o + ray.d*t;
                    const Vec3 L = light.c - pi;
                    const Vec3 N = sphere2.getNormal(pi);
                    float dt = dot(L.normalize(), N.normalize());
                    dt = fmax(dt,0.01);
                    if(!s1) {
                        pix_col = (blue + white*dt) * 0.5;
                        clamp255(pix_col);
                    } else if(t > told){
                        pix_col = ((pix_col) + ((blue + white*dt) * 0.5))/2;
                        clamp255(pix_col);
                    }
                }
                bool s3;
                float told2 = t;
                if(s3 = sphere3.intersect(ray, t)){
                    const Vec3 pi = ray.o + ray.d*t;
                    const Vec3 L = light.c - pi;
                    const Vec3 N = sphere2.getNormal(pi);
                    float dt = dot(L.normalize(), N.normalize());
                    dt = fmax(dt,0.01);
                    if(!s1 && !s2) {
                        pix_col = (blue + white*dt) * 0.5;
                        clamp255(pix_col);
                    } else if(t > told){
                        pix_col = ((pix_col) + ((blue + white*dt) * 0.5))/3;
                        clamp255(pix_col);
                    }
                }
                /*
                if(!s1 && !s2) {
                    if(!(x % 4)) {
                        lastfail = true;
                        x += 3;
                        for(int i = 0; i < 4; i++) {
                            fb[(y*W*3) + ((x+i)*3)] = (int)black.x;
                            fb[(y*W*3) + ((x+i)*3) + 1] = (int)black.y;
                            fb[(y*W*3) + ((x+i)*3) + 2] = (int)black.z;
                        }
                        //std::memset(fb + (y*W*3) + (x*3), 0, 4*3);
                        continue;
                    }
                } else {
                   if(lastfail) {
                        x -= 4;
                        lastfail = false;
                        continue;
                   }         
                }
                */
                fb[(y*W*3) + (x*3)] = (int)pix_col.x;
                fb[(y*W*3) + (x*3) + 1] = (int)pix_col.y;
                fb[(y*W*3) + (x*3) + 2] = (int)pix_col.z;
            }
        }
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(0,0);
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
        sphere2 = Sphere(Vec3(W*0.5+cos(frame/16.0)*100, (H*0.5)+cos(frame/16.0)*100, 50+sin(frame/16.0)*100), 20);
        sphere3 = Sphere(Vec3((W)*0.5+cos(frame/18.0)*-400, (H*0.5)-300/*+cos(frame/16.0)*100*/, 20/*+sin(frame/16.0)*100)*/), 60);
        //light = Sphere(Vec3((((sin(frame/32.0)/3.14159265))*W), ((sin(frame/32.0)/3.14159265))*H, 50), 1);
    }
}
