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

GLFWwindow *window;

bool firstrun = true;
Sphere *s;
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

int PrepFrameTest(Scene *man, Framebuffer &fb) {
    if(firstrun) {
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
        reflect.reflective = true;
        reflect.color = glm::vec3(0.0, 0.7, 0.0);
        Material white_unreflective;
        white_unreflective.color = glm::vec3(1.0, 1.0, 1.0);
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
        g.color = glm::vec3(0.0, 1.0, 0.0);
        b.color = glm::vec3(0.0, 0.0, 1.0);
        b.reflective = true;
        s = new Sphere(glm::vec3(0.0, 0.0, 0.0), 2, b);
        Sphere *s2 = new Sphere(glm::vec3(10.0, 0.0, 0.0), 2, g);
        Sphere *s3 = new Sphere(glm::vec3(10.0, 0.0, 10.0), 2, r);
        Light *l = new Light(glm::vec3(-10.0, 8.0, -10.0), glm::vec3(1.0, 1.0, 1.0), glm::vec2(1.0, 0.20));
        Light *l2 = new Light(glm::vec3(10, 10, 10), glm::vec3(0.8, 0.8, 0.8), glm::vec2(1.0, 0.20));
        man->AddObject(s);
        man->AddObject(s2);
        man->AddObject(s3);
        man->AddObject(tri);
        man->AddObject(tri2);
        man->AddObject(triR1);
        man->AddObject(triR2);
        man->AddObject(triR2);
        man->AddLight(l);
        man->AddLight(l2);
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
    //man.RegenerateObjectPositions();
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
    //cfg.lookat = glm::vec3(cos(vertical) * sin(horizontal), sin(vertical), cos(horizontal) * cos(vertical));
    glm::vec3 right = glm::vec3(sin(horizontal - 3.14f / 2.0f), 0, cos(horizontal - 3.14f / 2.0f));
    //cfg.up = glm::cross(right, cfg.lookat);
    cfg.up = glm::vec3(0.0, 1.0, 0.0);
    cfg.lookat = glm::normalize(glm::vec3(mat[0][3], 0.0, mat[2][3]) - cfg.center);
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
        mlocked = !mlocked;
    }
    if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        currentbackend = (currentbackend == Embree) ? Rendertape : Embree;
        man->SwitchBackend(currentbackend);
    }
    //cudaGetLastError();
    man->SetCameraConfig(cfg);
    return 0;
}

void DrawFrameTest(Scene *t, Framebuffer &fb) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 100, 0.0, 100, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_LIGHTING);
    glColor3f(1,1,1);
    glBindTexture(GL_TEXTURE_2D, fb.textureid);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
    glTexCoord2f(0, 1); glVertex3f(0, 100, 0);
    glTexCoord2f(1, 1); glVertex3f(100, 100, 0);
    glTexCoord2f(1, 0); glVertex3f(100, 0, 0);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glPopMatrix();


    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glDeleteTextures(1, &fb.textureid);
    //glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0,0);
    //glDrawPixels(fb.x, fb.y, GL_RGB, GL_UNSIGNED_BYTE, fb.fb);
    glColor3f(1.0, 1.0, 1.0);
    glRasterPos2i(0, 97);
    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)(std::string("Location: ") + StringifyVec3(cfg.center)).c_str());
    glRasterPos2i(0, 94);
    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)(std::string("Looking at: ") + StringifyVec3(cfg.lookat)).c_str());
    glRasterPos2i(0, 91);
    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)(std::string("Using backend: ") + BackendName[currentbackend]).c_str());
    glRasterPos2i(0, 88);
    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)("t - " + std::to_string(lastfps) + " FPS").c_str());
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
