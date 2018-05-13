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
#include <smmintrin.h>
#include <embree3/rtcore.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <GL/freeglut.h>
#include "triangle.h"
#include "sphere.h"
#include "vectorutil.h"
#include "scene.h"
#include <glm/gtc/matrix_transform.hpp>
#include <player.h>
#include "libfont.h"
extern "C" {
    #include "qdbmp.h"
}
GLFWwindow *window;

bool firstrun = true;
Sphere *s;
Sphere *s3;
Triangle *triE;
Triangle *triE2;
Triangle *triE3;
Triangle *triE4;
Triangle *triER;
Triangle *triER2;
int frame = 0;
int prevframe = frame;
float horizontal = 3.14f;
float vertical = 0.0f;
float mspeed = 0.005f;
float rotangle = 0.0f;
int lastfps;
bool mlocked = true;
timeval past, present;
CameraConfig cfg;
float lasttime;
RenderBackend currentbackend = OpenCL;
float s3velocity = 0.0;
float s3y = 3.0;
float dtextpos = 96;
int lasttrow = -1;
bool photo = false;
struct Text {
    Text() : transform(1.0f){
    }
    bool fade = false;
    double start;
    double end;
    glm::mat4x4 transform;
    std::string text;
    float x, y;
};
float lastphysrun = 0;
std::fstream f("lemonade.mod", std::ios_base::in | std::ios_base::binary);
ModulePlayer player(f);

std::vector<Text> texts;
std::vector<Sphere*> funspheres;

void playmodule() {
    //player.playModule();
}

float vectorstringoffset(std::string s) {
    float size = s.size()/2.0;
    return -(size*9.0 / 800)*100;
}


int PrepFrameTest(Scene *man, Framebuffer &fb) {
    if(firstrun) {
        std::thread t(&playmodule);
        t.detach();
        lasttime = glfwGetTime();
        firstrun = false;
        cfg.center = glm::vec3(12.0, 4.0, -36.0);
        printVec3(cfg.center);
        cfg.lookat  = glm::normalize(glm::vec3(0.0, 0.0, 0.0) - cfg.center);
        printVec3(cfg.lookat);
        cfg.up   = glm::vec3(1.0, 0.0, 0.0);
        printVec3(cfg.up);
        man->SetCameraConfig(cfg);
        Material reflect;
        reflect.type = REFLECT;
        reflect.color = glm::vec3(0.0, 0.7, 0.0);
        reflect.diffc = 1.0;
        reflect.specexp = 0;
        reflect.specc = 0;
        reflect.rindex = 1.0;
        Material white_unreflective;
        white_unreflective.color = glm::vec3(1.0, 1.0, 1.0);
        white_unreflective.diffc = 1.0;
        white_unreflective.specexp = 0.0;
        white_unreflective.specc = 0;
        white_unreflective.rindex = 1.0;
        Material white_reflective;
        white_reflective.color = glm::vec3(1.0, 1.0, 1.0);
        white_reflective.diffc = 1.0;
        white_reflective.specexp = 0.0;
        white_reflective.specc = 0;
        white_reflective.rindex = 1.31;
        white_reflective.type = REFLECT_REFRACT;
        glm::vec3 tris[3] = {glm::vec3(40.0, -3.0, -40.0), glm::vec3(-40.0, -3.0, 40.0), glm::vec3(-40.0, -3.0, -40.0)};
        Triangle *tri = new Triangle(tris, white_unreflective);
        glm::vec3 tris2[3] = {glm::vec3(40.0, -3.0, -40.0), glm::vec3(-40.0, -3.0, 40.0), glm::vec3(40.0, -3.0, 40.0)};
        Triangle *tri2 = new Triangle(tris2, white_unreflective);
        glm::vec3 trisR1[3] = {glm::vec3(-3.0, 0.25, 15.0), glm::vec3(-3.0, 0.25, -5.0), glm::vec3(-3.0, 8.0, 15.0)};
        Triangle *triR1 = new Triangle(trisR1, reflect);
        glm::vec3 trisR2[3] = {glm::vec3(-3.0, 0.25, -5.0), glm::vec3(-3.0, 8.0, -5.0), glm::vec3(-3.0, 8.0, 15.0)};
        Triangle *triR2 = new Triangle(trisR2, reflect);
        Material r, g, b;
        r.color = glm::vec3(1.0, 0.0, 0.0);
        r.type = REFLECT;
        g.color = glm::vec3(0.0, 1.0, 0.0);
        b.color = glm::vec3(0.0, 0.0, 1.0);
        b.type = DIFFUSE_GLOSS;
        b.diffc = 0.6;
        b.specc = 0.4;
        b.specexp = 0.05;
        g.type = DIFFUSE_GLOSS;
        g.diffc = 0.3;
        g.specc = 0.7;
        g.specexp = 0.1;
        g.emits = 0.0;
        Material emissive;
        emissive.color = glm::vec3(0.456, 0.456, 1.0);
        emissive.type = DIFFUSE_GLOSS;
        emissive.alightid = 1;
        emissive.specexp = 0.0;
        emissive.diffc = 1.0;
        emissive.specc = 0.0;
        emissive.emits = 1.0;
        emissive.rindex = 1.0;
        Material greenemits;
        greenemits.color = glm::vec3(0, 0.5,0.33);
        //greenemits.color = glm::vec3(0.103634, 0.6253447208, 0.9573695762);
        greenemits.type = DIFFUSE_GLOSS;
        greenemits.alightid = 1;
        greenemits.specexp = 0.0;
        greenemits.diffc = 1.0;
        greenemits.specc = 0.0;
        greenemits.emits = 1.0;
        greenemits.rindex = 1.0;
        greenemits.alightid = 2;
        Material redemits;
        redemits.color = glm::vec3(0.5,0.0,0.1);
        //redemits.color = glm::vec3(0.91575012927, 0.40454082256, 0.48776520187);
        redemits.type = DIFFUSE_GLOSS;
        redemits.alightid = 1;
        redemits.specexp = 0.0;
        redemits.diffc = 1.0;
        redemits.specc = 0.0;
        redemits.emits = 1.0;
        redemits.rindex = 1.0;
        redemits.alightid = 3;
        glm::mat4x4 m;
        glm::vec3 etris[3] = {glm::vec3(1.5, 1.5, -2.0), glm::vec3(1.5, 1.5, -5.0), glm::vec3(3, 4.5, -2.0)};
        glm::vec3 etris2[3] = {glm::vec3(1.5, 1.5, -5.0), glm::vec3(3, 4.5, -5.0), glm::vec3(3, 4.5, -2.0)};
        glm::vec3 etrisx[3] = {glm::vec3(8, -3.5, 2.0), glm::vec3(8, -3.5, 5.0), glm::vec3(8, -7.5, 2.0)};
        glm::vec3 etrisx2[3] = {glm::vec3(8, -3.5, 5.0), glm::vec3(8, -7.5, 5.0), glm::vec3(8, -7.5, 2.0)};
        glm::vec3 etrisr[3] = {glm::vec3(-10.5, 1.5, 5.0), glm::vec3(-10.5, 1.5, 2.0), glm::vec3(-9.0, 4.5, 5.0)};
        glm::vec3 etrisr2[3] = {glm::vec3(-10.5, 1.5, 2.0), glm::vec3(-9.0, 4.5, 2.0), glm::vec3(-9.0, 4.5, 5.0)};
        triE = new Triangle(etris, emissive);
        triE2 = new Triangle(etris2, emissive);
        glm::vec3 etris3[3] = {glm::vec3(etrisx[0].y, etrisx[0].x, etrisx[0].z), glm::vec3(etrisx[1].y, etrisx[1].x, etrisx[1].z), glm::vec3(etrisx[2].y, etrisx[2].x, etrisx[2].z)};
        glm::vec3 etris4[3] = {glm::vec3(etrisx2[0].y, etrisx2[0].x, etrisx2[0].z), glm::vec3(etrisx2[1].y, etrisx2[1].x, etrisx2[1].z), glm::vec3(etrisx2[2].y, etrisx2[2].x, etrisx2[2].z)};
        triE3 = new Triangle(etris3, greenemits);
        triE4 = new Triangle(etris4, greenemits);
        triER = new Triangle(etrisr, redemits);
        triER2 = new Triangle(etrisr2, redemits);
        s = new Sphere(glm::vec3(0.0, 0.0, 0.0), 2, b);
        Sphere *s2 = new Sphere(glm::vec3(10.0, 0.0, 0.0), 2, g);
        s3 = new Sphere(glm::vec3(10.0, 3.0, 10.0), 4, r);
        Light *l = new Light(glm::vec3(-10.0, 8.0, -10.0), glm::vec3(0.0, 0.0, 1.0), glm::vec2(1.0, 0.20));
        Light *l2 = new Light(glm::vec3(10, 20, 10), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 0.20));
        Sphere *golfball = new Sphere(glm::vec3(-5.5, 1, 2.0), 2, white_reflective);
//        for(int i = 0; i < 10; i++) {
//            Material gdif;
//            gdif.type = DIFFUSE_GLOSS;
//            gdif.diffc = 0.9;
//            gdif.specc = 0.1;
//            gdif.specexp = 0.1;
//            gdif.color = glm::vec3(0.0, 1.0, 0.0);

//            Sphere *gs = new Sphere(glm::vec3(0, 5, 25), 2, gdif);
//            funspheres.push_back(gs);
//            man->AddObject(gs);
//        }
//        for(int i = 0; i < 10; i++) {
//            Material gdif;
//            gdif.type = DIFFUSE_GLOSS;
//            gdif.diffc = 0.9;
//            gdif.specc = 0.1;
//            gdif.specexp = 0.1;
//            gdif.color = glm::vec3(0.0, 1.0, 0.0);
//            auto floatrand = [&]() {
//                return (man->fast_rand()/16384.0)-1.0;
//            };

//            glm::vec3 trisRg[3] = {glm::vec3(floatrand()*100.0, floatrand()*100.0, floatrand()*100.0), glm::vec3(floatrand()*100.0, floatrand()*100.0, floatrand()*100.0), glm::vec3(floatrand()*100.0, floatrand()*100.0, floatrand()*100.0)};
//            Triangle *triRg = new Triangle(trisRg, gdif);
//            man->AddObject(triRg);
//        }
        man->AddObject(s);
        man->AddObject(s2);
        man->AddObject(s3);
        man->AddObject(tri);
        man->AddObject(tri2);
        man->AddObject(triR1);
        man->AddObject(triR2);
        man->AddObject(triE);
        man->AddObject(triE2);
        man->AddObject(triE3);
        man->AddObject(triE4);
        man->AddObject(triER);
        man->AddObject(triER2);
        man->AddObject(golfball);
        //        for(int i = 0; i < 20; i++) {
        //            man->AddObject(triR1);
        //            man->AddObject(triR2);
        //        }
        //man->AddLight(l);
        //man->AddLight(l2);
        srand(0);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, (GLfloat) 100, 0.0, (GLfloat) 100);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gettimeofday(&past, NULL);
        glfwSetCursorPos(window, fb.x/2, fb.y/2);
        glfwSetInputMode(window, GLFW_CURSOR,GLFW_CURSOR_HIDDEN);
        glPixelStorei(GL_PACK_ALIGNMENT, 8);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    }
    float time = glfwGetTime();
//    for(int i = 0; i < 10; i++) {
//        glm::mat4x4 a = funspheres[i]->getTransform();
//        a[3][1] = 5 + 5*sin(time + i*0.2 );
//        a[3][0] = 20*sin(-1.6 + time + 0.2*i);
//        a[3][2] = 25*sin( 0.5 + time + i*0.4);
//        funspheres[i]->setTransform(a);
//    }
    if(glfwWindowShouldClose(window))
        return -1;
    float tdiff = (glfwGetTime() - lasttime)*32;
    lasttime = glfwGetTime();
    glm::mat4x4 mat;
    mat = glm::translate(mat, glm::vec3(sin(glfwGetTime()/2.0)*20, 0.0, cos(glfwGetTime()/2.0)*20));
    s->setTransform(mat);
    int w,h;
    glfwGetWindowSize(window, &w, &h);
    double xpos = w/2, ypos = h/2;
    if(mlocked) {
        glfwGetCursorPos(window, &xpos, &ypos);
        glfwSetCursorPos(window, w/2, h/2);
        horizontal += mspeed * -(w/2- xpos);
        vertical += mspeed * (h/2 - ypos);
    }

    if (vertical > 1.5f) {
        vertical = 1.5f;
    }
    else if (vertical < -1.5f) {
        vertical = -1.5f;
    }
    cfg.lookat = glm::vec3(cos(vertical) * sin(horizontal), sin(vertical), cos(horizontal) * cos(vertical));
    glm::vec3 right = glm::vec3(sin(horizontal - 3.14f / 2.0f) * cos(rotangle), sin(rotangle), cos(horizontal - 3.14f / 2.0f) * cos(rotangle));
    cfg.up = glm::cross(right, cfg.lookat);
    if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cfg.center += cfg.lookat*tdiff;
    }
    if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cfg.center -= cfg.lookat*tdiff;
    }
    if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cfg.center += right*tdiff;
    }
    if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cfg.center -= right*tdiff;
    }
    if(glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        rotangle -= (1/90.0f) * (3.14/2.0f) * tdiff;
    }
    if(glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        rotangle += (1/90.0f) * (3.14/2.0f) * tdiff;
    }
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        return -1;
    }
    if(glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
        photo = true;
    }
    man->SetCameraConfig(cfg);
    return 0;
}

void DrawFrameTest(Scene *t, Framebuffer &fb) {
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 100, 0.0, 100, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_LIGHTING);
    glColor3f(1,1,1);
    glColor3f(1.0, 1.0, 1.0);
    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, fb.textureid);
    if(/*glfwGetTime() > 8.0*/ true) {
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
        glTexCoord2f(0, 1); glVertex3f(0, 100, 0);
        glTexCoord2f(1, 1); glVertex3f(100, 100, 0);
        glTexCoord2f(1, 0); glVertex3f(100, 0, 0);
        glEnd();
    }
    if(photo) {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        BMP *bmp;
        photo = false;
        GLubyte* pixels = (GLubyte*) malloc(fb.x * fb.y * 4 * sizeof(GLubyte));
        glReadPixels(0, 0, fb.x, fb.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        char *out_fn = "screenie.bmp";
        GLubyte *p = pixels;
        bmp = BMP_Create(fb.x, fb.y, 24);
        if(!bmp) {
            return;
        }
        for(uint64_t x = 0; x < fb.x; x++){
            for(uint64_t y = 0; y < fb.y; y++) {
                BMP_SetPixelRGB(bmp, x, y, p[((y * fb.x)*4) + (x * 4)],
                                           p[((y * fb.x)*4) + (x * 4) + 1],
                                           p[((y * fb.x)*4) + (x * 4) + 2]);
            }
        }
        printf("Writing BMP to %s\n", out_fn);
        BMP_WriteFile(bmp, out_fn);
    }
    float time = glfwGetTime();
    glDeleteTextures(1, &fb.textureid);
    //glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0,0);
    //glDrawPixels(fb.x, fb.y, GL_RGB, GL_UNSIGNED_BYTE, fb.fb);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for(size_t i = 0; i < texts.size(); i++) {
        if((time >texts[i].start) && (time < texts[i].end)) {
            GLuint tex;
            uint8_t *r = drawText(texts[i].text, glm::vec4(0.0, 0.0, 0.0, 1));
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texts[i].text.size()*9, 15, 0, GL_RGBA, GL_UNSIGNED_BYTE, r);
            glBindTexture(GL_TEXTURE_2D, tex);
            glBegin(GL_QUADS);
            float xsz = ((texts[i].text.size()*9)/(float)fb.x)*100;
            glm::vec4 v1(texts[i].x, (texts[i].y - (7.5/fb.y)*100), 0, 1);
            glm::vec4 v2(texts[i].x, (texts[i].y + (7.5/fb.y)*100), 0, 1);
            glm::vec4 v3((texts[i].x + xsz), (texts[i].y + (7.5/fb.y)*100), 0, 1);
            glm::vec4 v4((texts[i].x + xsz), (texts[i].y - (7.5/fb.y)*100), 0, 1);
            glm::vec4 vorigin((texts[i].x + xsz/2), texts[i].y, 0, 1);
            v1 = (texts[i].transform * (v1 - vorigin)) + vorigin ;
            v2 = (texts[i].transform * (v2 - vorigin)) + vorigin ;
            v3 = (texts[i].transform * (v3 - vorigin)) + vorigin ;
            v4 = (texts[i].transform * (v4 - vorigin)) + vorigin ;
            for(int a = 0; a < 1; a++) {
                glTexCoord2f(0, 0); glVertex3f(v1.x, v1.y, 0);
                glTexCoord2f(0, 1); glVertex3f(v2.x, v2.y, 0);
                glTexCoord2f(1, 1); glVertex3f(v3.x, v3.y, 0);
                glTexCoord2f(1, 0); glVertex3f(v4.x, v4.y, 0);
            }
            glEnd();
            glDeleteTextures(1, &tex);
            delete[] r;
        }
    }
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glfwSwapBuffers(window);
    glfwPollEvents();
    gettimeofday(&present, NULL);
    if(present.tv_sec > past.tv_sec) {
        int fps = (frame - prevframe);
        std::string title = "t - " + std::to_string(fps) + " FPS";
        prevframe = frame;
        lastfps = fps;
        glfwSetWindowTitle(window, title.c_str());
        past = present;
    }
    frame++;
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    glfwInit();
    window = glfwCreateWindow(1280, 720, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    Scene man(currentbackend, 4, PrepFrameTest, DrawFrameTest);
    Framebuffer fb;
    fb.x = 1280;
    fb.y = 720;
    fb.fb = (uint8_t*)malloc(fb.x*fb.y*3);

    //cudaDeviceSynchronize();
    man.render(fb);
}
