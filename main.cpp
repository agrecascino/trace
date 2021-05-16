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
#include "bmpread.h"
#include <glm/gtx/transform.hpp>
#include <glm/ext/matrix_projection.hpp>
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
RenderBackend currentbackend = (RenderBackend)5;
float s3velocity = 0.0;
float s3y = 3.0;
float dtextpos = 96;
int lasttrow = -1;
bmpread_t bmp;
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
std::fstream f("encyclopaedia-part2.mod", std::ios_base::in | std::ios_base::binary);
ModulePlayer player(f);

std::vector<Text> texts;

void playmodule() {
    player.playModule();
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



float cos2(float x) {
    float pi2 = 3.1415926535897932*2;
    return (sine_wave[(uint8_t)((fmod(x+ pi2/4.0, 2*pi2)/pi2)*255)]/127.5)-1.0;
}

float sin2(float x) {
    float pi2 = 3.1415926535897932*2;
    return (sine_wave[(uint8_t)((fmod(x, 2*pi2)/pi2)*255)]/127.5)-1.0;
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
        assert(bmpread("example.bmp", BMPREAD_ANY_SIZE, &bmp));
        struct Text predestination;
        predestination.start = 0;
        predestination.end = 8;
        predestination.text = "insert text here";
        predestination.x = -vectorstringoffset(predestination.text)*2 + 1;
        predestination.y = 96;
        predestination.transform = glm::mat4x4(2.0f);
        texts.push_back(predestination);
        struct Text echo;
        echo.start = 0;
        echo.end = 8;
        echo.text = "anyone out there";
        echo.x = -vectorstringoffset(echo.text)*2 + 1;
        echo.y = 4;
        echo.transform = glm::mat4x4(2.0f);
        texts.push_back(echo);
        struct Text start;
        start.start = 0;
        start.end = 2;
        start.text = "name presents";
        start.x = vectorstringoffset(start.text) + 50;
        start.y = 50;
        start.transform = glm::mat4x4(2.0f);
        texts.push_back(start);
        struct Text name;
        name.start = 2;
        name.end = 5;
        name.text = "something";
        name.x = vectorstringoffset(name.text) + 50;
        name.y = 50;
        name.transform = glm::mat4x4(2.0f);
        texts.push_back(name);
        struct Text multimedia;
        multimedia.start = 18;
        multimedia.end = 28;
        multimedia.text = "CPU is multimedia";
        multimedia.x = vectorstringoffset(multimedia.text) + 50;
        multimedia.y = 96;
        multimedia.transform = glm::mat4x4(2.0f);
        texts.push_back(multimedia);
        struct Text creativity;
        creativity.start = 18;
        creativity.end = 28;
        creativity.text = "CPU is creativity";
        creativity.x = vectorstringoffset(creativity.text) + 50;
        creativity.y = 4;
        creativity.transform = glm::mat4x4(2.0f);
        texts.push_back(creativity);

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
    if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        currentbackend = (currentbackend == Embree) ? Rendertape : Embree;
        man->SwitchBackend(currentbackend);
    }
    float t = glfwGetTime();
    if(t < 5 && t > 2) {
        std::string s[] = { "something" , "a horrible demo", "inspired by sunflower", " a disaster piece" };
        texts[3].text = s[(int)(rand()) % 4];
        texts[3].x = vectorstringoffset(texts[3].text) + 50;
    }
    if(t > 17.9) {
        float hi = (exp(t-19.0f)- 1.0);
        texts[4].x = vectorstringoffset(texts[4].text) + 57 * std::min(hi, 1.0f) - 7;
        texts[5].x = vectorstringoffset(texts[4].text) + 57 * std::min(hi, 1.0f) - 7;

    }
    glm::mat4x4 transform;
    transform[0][0] = cos(t/8.0f);
    transform[0][1] = sin(t/8.0f);
    transform[1][0] = -sin(t/8.0f);
    transform[1][1] = cos(t/8.0f);
    glm::vec2 center;
    center.x = sin(t)*800 + 800 + sin((t/16.0))*200;
    center.y = cos(t)*450 + 450 + cos(t/16.0 + 1.90)*200;
    float factor = 1/std::max(1.0f, (t-11.0f));
    float factor2 = std::min(1.0f, ((t-16.0f)/2.0f)*((t-16.0f)/3.0f)) * ((cos(std::max(1.0f, t-17.0f))+1.0f)/2.0f + 0.5f) * exp((t-18.0f)/3.5f);
    float atten = std::max(1.0f, (t-12.0f));
    if(t < 16.0) {
        for(size_t x = 0; x < fb.x; x++) {
            for(size_t y = 0; y < fb.y; y++) {
                float xdist =center.x - (x*factor);
                xdist *= xdist;
                float ydist =center.y - (y*factor);
                ydist *= ydist;
                float dist = sqrtf(xdist + ydist)/65536.0f;
                fb.fb[(y*1600*3) + x*3] = std::min((1.0/(dist)) * (cos(t/2.0f + 2) +1.0f)/2.0f, 255.0);
                fb.fb[(y*1600*3) + x*3 + 1] = std::min((1.0/(dist))* (cos(t/2.0f + 9) +1.0f)/2.0f, 255.0);
                fb.fb[(y*1600*3) + x*3 + 2] = std::min((1.0/(dist))* (cos(t/2.0f + 5) +1.0f)/2.0f, 255.0);
            }
        }
    } else if(t < 32.0) {
#pragma omp parallel for
        for(size_t x = 0; x < fb.x; x++) {
            for(size_t y = 0; y < fb.y; y++) {
                float xadj = (((int)x - 800)/64.0)*(factor2) + 20*sin(t/128.0f);
                float yadj = (((int)y - 450)/64.0)*(factor2) + 10*cos(t/128.0f);
                glm::vec4 rotated(xadj, yadj, 0, 1);
                rotated = transform * rotated;
                xadj = rotated.x;
                yadj = rotated.y;
                const float amp = sin2(-0.8 + sin2(xadj+ t*0.5) + sin2(yadj+t*.075) + t*1.5 + cos2(xadj+yadj+t*1.4))+1.0;
                fb.fb[(y*1600*3) + x*3] = amp * 16;
                fb.fb[(y*1600*3) + x*3 + 1] = amp * 8;
                fb.fb[(y*1600*3) + x*3 + 2] = amp * 64;
            }
        }
    } else if(t < 30) {
        glm::mat4x4 view = glm::lookAt(glm::vec3(-1, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        //glm::mat4x4 projection = glm::project(45.0f, 16/9.0f, 0.01f, 10.0f);

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
    glPopMatrix();


    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glDeleteTextures(1, &fb.textureid);
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0,0);
    glDrawPixels(fb.x, fb.y, GL_RGB, GL_UNSIGNED_BYTE, fb.fb);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, 100, 0.0, 100, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glPushMatrix();
    glLoadIdentity();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for(size_t i = 0; i < texts.size(); i++) {
        if((time >texts[i].start) && (time < texts[i].end)) {
            GLuint tex;
            uint8_t *r = drawText(texts[i].text, glm::vec4(0.9, 0.9, 0.9, 0.8));
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
                float yoff = 0;
                float xoff = 0;
                glTexCoord2f(0, 0); glVertex3f(v1.x + xoff, v1.y + yoff, 0);
                glTexCoord2f(0, 1); glVertex3f(v2.x + xoff, v2.y + yoff, 0);
                glTexCoord2f(1, 1); glVertex3f(v3.x + xoff, v3.y + yoff, 0);
                glTexCoord2f(1, 0); glVertex3f(v4.x + xoff, v4.y + yoff, 0);
            }
            glEnd();
            glDeleteTextures(1, &tex);
            delete[] r;
        }
    }
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
