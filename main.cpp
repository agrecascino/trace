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
    double start;
    double end;
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
    return -(size*9.0 / 1600)*100;
}


uint8_t  sine_wave[256] = {
  0x80, 0x83, 0x86, 0x89, 0x8C, 0x90, 0x93, 0x96,
  0x99, 0x9C, 0x9F, 0xA2, 0xA5, 0xA8, 0xAB, 0xAE,
  0xB1, 0xB3, 0xB6, 0xB9, 0xBC, 0xBF, 0xC1, 0xC4,
  0xC7, 0xC9, 0xCC, 0xCE, 0xD1, 0xD3, 0xD5, 0xD8,
  0xDA, 0xDC, 0xDE, 0xE0, 0xE2, 0xE4, 0xE6, 0xE8,
  0xEA, 0xEB, 0xED, 0xEF, 0xF0, 0xF1, 0xF3, 0xF4,
  0xF5, 0xF6, 0xF8, 0xF9, 0xFA, 0xFA, 0xFB, 0xFC,
  0xFD, 0xFD, 0xFE, 0xFE, 0xFE, 0xFF, 0xFF, 0xFF,
  0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFE, 0xFE, 0xFD,
  0xFD, 0xFC, 0xFB, 0xFA, 0xFA, 0xF9, 0xF8, 0xF6,
  0xF5, 0xF4, 0xF3, 0xF1, 0xF0, 0xEF, 0xED, 0xEB,
  0xEA, 0xE8, 0xE6, 0xE4, 0xE2, 0xE0, 0xDE, 0xDC,
  0xDA, 0xD8, 0xD5, 0xD3, 0xD1, 0xCE, 0xCC, 0xC9,
  0xC7, 0xC4, 0xC1, 0xBF, 0xBC, 0xB9, 0xB6, 0xB3,
  0xB1, 0xAE, 0xAB, 0xA8, 0xA5, 0xA2, 0x9F, 0x9C,
  0x99, 0x96, 0x93, 0x90, 0x8C, 0x89, 0x86, 0x83,
  0x80, 0x7D, 0x7A, 0x77, 0x74, 0x70, 0x6D, 0x6A,
  0x67, 0x64, 0x61, 0x5E, 0x5B, 0x58, 0x55, 0x52,
  0x4F, 0x4D, 0x4A, 0x47, 0x44, 0x41, 0x3F, 0x3C,
  0x39, 0x37, 0x34, 0x32, 0x2F, 0x2D, 0x2B, 0x28,
  0x26, 0x24, 0x22, 0x20, 0x1E, 0x1C, 0x1A, 0x18,
  0x16, 0x15, 0x13, 0x11, 0x10, 0x0F, 0x0D, 0x0C,
  0x0B, 0x0A, 0x08, 0x07, 0x06, 0x06, 0x05, 0x04,
  0x03, 0x03, 0x02, 0x02, 0x02, 0x01, 0x01, 0x01,
  0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x03,
  0x03, 0x04, 0x05, 0x06, 0x06, 0x07, 0x08, 0x0A,
  0x0B, 0x0C, 0x0D, 0x0F, 0x10, 0x11, 0x13, 0x15,
  0x16, 0x18, 0x1A, 0x1C, 0x1E, 0x20, 0x22, 0x24,
  0x26, 0x28, 0x2B, 0x2D, 0x2F, 0x32, 0x34, 0x37,
  0x39, 0x3C, 0x3F, 0x41, 0x44, 0x47, 0x4A, 0x4D,
  0x4F, 0x52, 0x55, 0x58, 0x5B, 0x5E, 0x61, 0x64,
  0x67, 0x6A, 0x6D, 0x70, 0x74, 0x77, 0x7A, 0x7D
};

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
        white_unreflective.rindex = 1.8;
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
        b.specexp = 25;
        g.type = DIFFUSE_GLOSS;
        g.diffc = 0.3;
        g.specc = 0.7;
        g.specexp = 20;
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
        struct Text intro;
        intro.text = "anon presents";
        intro.start = 0;
        intro.end = 4;
        intro.x = vectorstringoffset(intro.text) + 50;
        intro.y = 50;
        struct Text intro2;
        intro2.text = "welcome to cs 148: computer graphics and imaging";
        intro2.start = 9;
        intro2.end = 18;
        intro2.x = vectorstringoffset(intro2.text) + 50;
        intro2.y = 85;
        struct Text text;
        text.text = "we control the translate";
        text.start = 20;
        text.end = 30;
        text.x = 80;
        text.y = 96;
        struct Text text2;
        text2.text = "we control the rotate";
        text2.start = 22.5;
        text2.end = 30;
        text2.x = 80;
        text2.y = 92;
        struct Text t3;
        t3.text = "sinusoid love, from C to you.";
        t3.start = 0.0;
        t3.end = 2321312312;
        t3.x = vectorstringoffset(t3.text) + 50;
        t3.y = 50;
        texts.push_back(t3);
//        texts.push_back(text);
//        texts.push_back(text2);
//        texts.push_back(intro);
//        texts.push_back(intro2);
    }
    texts[0].y = 50 + 5*sin(1.5*glfwGetTime());
    if(glfwWindowShouldClose(window))
        return -1;
    float tdiff = (glfwGetTime() - lasttime)*32;
    lasttime = glfwGetTime();
    glm::mat4x4 mat;
    mat = glm::translate(mat, glm::vec3(sin(glfwGetTime()/2.0)*20, 0.0, cos(glfwGetTime()/2.0)*20));
    s->setTransform(mat);
    //man.RegenerateObjectPositions();
    double xpos = fb.x/2, ypos = fb.y/2;
//    if((player.lastorder == 1) && !(player.lastrow % 8) && (lasttrow != player.lastrow)) {
//        Text fun;
//        fun.text = "WARNING: THIS DEMONSTRATION RUNS IN REALTIME";
//        fun.start = glfwGetTime();
//        fun.end = 7.98;
//        fun.x = vectorstringoffset("WARNING: THIS DEMONSTRATION RUNS IN REALTIME") + 50;
//        fun.y = dtextpos;
//        dtextpos -= 3;
//        texts.push_back(fun);
//        lasttrow = player.lastrow;
//    }
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
    glm::mat4x4 s3mat = s3->getTransform();
    if((glfwGetTime()- 0.016) > lastphysrun) {
        s3velocity -= (-40/60.0)/2.0;
        s3y += s3velocity;
        //if(s3y < 0.0)
            //s3velocity = 0.20;
        if(!(player.lastrow % 8))
            s3y = 5.0;
        lastphysrun = glfwGetTime();
    }
    s3mat[3][1] = s3y;
    s3->setTransform(s3mat);
    //cfg.up = glm::vec3(0.0, 1.0, 0.0);
    //cfg.center = glm::vec3(sin(glfwGetTime()/2.0)*80, 12.0, cos(glfwGetTime()/2.0)*80);
    //cfg.lookat = glm::normalize(glm::vec3(s3mat[3][0], 0.0, s3mat[3][2]) - cfg.center);
    //cfg.lookat = glm::normalize(glm::vec3(mat[3][0], 0.0, mat[3][2]) - cfg.center);
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
    if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        currentbackend = (currentbackend == Embree) ? Rendertape : Embree;
        man->SwitchBackend(currentbackend);
    }
    //cudaGetLastError();
    float t = glfwGetTime();
#pragma omp parallel for
    for(size_t x = 0; x < 1600; x++) {
        for(size_t y = 0; y < 900; y++) {
            uint8_t amp = (sine_wave[(x/4 + (int)(t*200)) & 0xff] + sine_wave[(y/4 + (int)(glfwGetTime()*50)) & 0xff] + (int)(glfwGetTime()*10)) + sine_wave[(x/4+y/4) & 0xff] + sine_wave[(int)(x/4+134*(t))&0xff];
            fb.fb[y*1600*3 + x*3] = ((uint16_t)amp * 16) >> 8;
            fb.fb[y*1600*3 + x*3 + 1] = ((uint16_t)amp * 32) >> 8;
            fb.fb[y*1600*3 + x*3 + 2] = ((uint16_t)amp * 64) >> 8;

        }
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
    glBindTexture(GL_TEXTURE_2D, fb.textureid);
    if(/*glfwGetTime() > 8.0*/ true) {
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex3f(0, 0, 0);
        glTexCoord2f(0, 1); glVertex3f(0, 100, 0);
        glTexCoord2f(1, 1); glVertex3f(100, 100, 0);
        glTexCoord2f(1, 0); glVertex3f(100, 0, 0);
        glEnd();
    }

    glDisable(GL_TEXTURE_2D);
    glPopMatrix();


    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glDeleteTextures(1, &fb.textureid);
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0,0);
    glDrawPixels(fb.x, fb.y, GL_RGB, GL_UNSIGNED_BYTE, fb.fb);
    glColor3f(1.0, 1.0, 1.0);
    float cur_time = glfwGetTime();
    for(Text t : texts) {
        float start = t.start;
        float end = t.end;
        if((start < cur_time) && end > cur_time) {
            glRasterPos2f(t.x, t.y);
            glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)t.text.c_str());
        }
    }
    //glRasterPos2i(0, 97);
//    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)(std::string("Location: ") + StringifyVec3(cfg.center)).c_str());
//    glRasterPos2i(0, 94);
//    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)(std::string("Looking at: ") + StringifyVec3(cfg.lookat)).c_str());
//    glRasterPos2i(0, 91);
//    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)(std::string("Using backend: ") + BackendName[currentbackend]).c_str());
//    glRasterPos2i(0, 88);
//    glutBitmapString(GLUT_BITMAP_9_BY_15, (unsigned char*)("t - " + std::to_string(lastfps) + " FPS").c_str());
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
