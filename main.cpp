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

GLFWwindow *window;

bool firstrun = true;
Sphere *s;
Sphere *s3;
int frame = 0;
int prevframe = frame;
float horizontal = 3.14f;
float vertical = 0.0f;
float mspeed = 0.005f;
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
        cfg.center = glm::vec3(12.0, 4.0, 12.0);
        printVec3(cfg.center);
        cfg.lookat  = glm::normalize(glm::vec3(0.0, 0.0, 0.0) - cfg.center);
        printVec3(cfg.lookat);
        cfg.up   = glm::vec3(1.0, 0.0, 0.0);
        printVec3(cfg.up);
        man->SetCameraConfig(cfg);
        Material reflect;
        reflect.type = REFLECT;
        reflect.color = glm::vec3(0.0, 0.7, 0.0);
        Material white_unreflective;
        white_unreflective.color = glm::vec3(1.0, 1.0, 1.0);
        white_unreflective.diffc = 1.0;
        white_unreflective.specexp = 0;
        white_unreflective.specc = 0;
        white_unreflective.rindex = 1.0;
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
        s = new Sphere(glm::vec3(0.0, 0.0, 0.0), 2, b);
        Sphere *s2 = new Sphere(glm::vec3(10.0, 0.0, 0.0), 2, g);
        s3 = new Sphere(glm::vec3(10.0, 3.0, 10.0), 4, r);
        Light *l = new Light(glm::vec3(-10.0, 8.0, -10.0), glm::vec3(1.0, 0.0, 1.0), glm::vec2(1.0, 0.20));
        Light *l2 = new Light(glm::vec3(10, 30, 10), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 0.20));
        man->AddObject(s);
        man->AddObject(s2);
        man->AddObject(s3);
        man->AddObject(tri);
        man->AddObject(tri2);
        man->AddObject(triR1);
        man->AddObject(triR2);
        //        for(int i = 0; i < 20; i++) {
        //            man->AddObject(triR1);
        //            man->AddObject(triR2);
        //        }
        man->AddLight(l);
        man->AddLight(l2);
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
    if(glfwWindowShouldClose(window))
        return -1;
    float tdiff = (glfwGetTime() - lasttime)*32;
    lasttime = glfwGetTime();
    glm::mat4x4 mat;
    mat = glm::translate(mat, glm::vec3(sin(glfwGetTime()/2.0)*20, 0.0, cos(glfwGetTime()/2.0)*20));
    s->setTransform(mat);
    double xpos = fb.x/2, ypos = fb.y/2;
    if(mlocked) {
        glfwGetCursorPos(window, &xpos, &ypos);
        glfwSetCursorPos(window, fb.x/2, fb.y/2);
        horizontal += mspeed * -(fb.x/2 - xpos);
        vertical += mspeed * (fb.y/2 - ypos);
    }

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
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        return -1;
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
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
    window = glfwCreateWindow(1600, 900, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    Scene man(currentbackend, 4, PrepFrameTest ,DrawFrameTest);
    Framebuffer fb;
    fb.x = 1600;
    fb.y = 900;
    fb.fb = (uint8_t*)malloc(fb.x*fb.y*3);

    //cudaDeviceSynchronize();
    man.render(fb);
}
