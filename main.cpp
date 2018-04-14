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
        assert(bmpread("example.bmp", BMPREAD_ANY_SIZE, &bmp));
        struct Text intro;
        intro.text = "name presents \129";
        intro.start = 0;
        intro.end = 4;
        intro.x = vectorstringoffset(intro.text) + 50;
        intro.transform = glm::mat4x4(3.0f);
        intro.y = 50;
        struct Text love;
        love.text = "why do you hurt me?";
        love.start = 0;
        love.end = 4;
        love.x = vectorstringoffset(love.text) + 50;
        love.transform = glm::mat4x4(2.5f);
        love.y = 20;
        struct Text love2;
        love2.text = "i just want to love you.";
        love2.start = 4;
        love2.end = 8;
        love2.x = vectorstringoffset(love.text) + 50;
        love2.transform = glm::mat4x4(2.5f);
        love2.y = 20;
        struct Text hello;
        hello.text = "who are you?";
        hello.start = 10;
        hello.end = 16;
        hello.x = vectorstringoffset(hello.text) + 50;
        hello.transform = glm::mat4x4(2.5f);
        hello.y = 50;
        hello.fade = true;
        struct Text intro2;
        intro2.text = "a new demo";
        intro2.start = 4;
        intro2.end = 8;
        intro2.x = vectorstringoffset(intro2.text) + 50;
        intro2.transform = glm::mat4x4(2.5f);
        intro2.y = 50;
        struct Text text;
        text.text = "WIDE SPECTRUM ULTRAVIOLET";
        text.start = 8;
        text.end = 15;
        text.x = vectorstringoffset(text.text) + 50;
        text.transform = glm::mat4x4(2.0f);
        text.y = 50;
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
        struct Text sync;
        sync.text = "synchronization l ost";
        sync.start = 8;
        sync.end = 10;
        sync.x = vectorstringoffset(sync.text) + 50;
        sync.y = 50;
        sync.transform = glm::mat4x4(1.5f);
        //        texts.push_back(sync);
        struct Text sync2;
        sync2.text = "synchronizationl ost";
        sync2.start = 8;
        sync2.end = 10;
        sync2.x = vectorstringoffset(sync2.text) + 50;
        sync2.y = 50;
        sync2.transform = glm::mat4x4(1.5f);
        texts.push_back(hello);
        texts.push_back(hello);
        texts.push_back(hello);
        //        texts.push_back(sync2);
        //texts.push_back(t3);
        //                texts.push_back(text);
        //        texts.push_back(text2);
        //                texts.push_back(intro);
        //                texts.push_back(intro2);
        texts.push_back(love);
        texts.push_back(love2);
    }

    //    if(glfwGetTime() > 8.0)
    //        texts[0].x = vectorstringoffset(texts[0].text) + 50 + (int)80*sin(8*glfwGetTime()*glfwGetTime());
    //    texts[0].y = 50 + rand() % 10;
    //    if(glfwGetTime() > 8.0)
    //        texts[1].x = vectorstringoffset(texts[1].text) + 50 + (int)80*sin(8*glfwGetTime()*glfwGetTime()+29);
    //    texts[1].y = 50 + rand() % 5;
    if(glfwGetTime() > 7.0) {
        texts[4].x = vectorstringoffset(texts[0].text) + 50 + (int)80*sin(8*glfwGetTime()*glfwGetTime());
        texts[4].y = 20.0 + (rand() % 5);
    } else {
        texts[3].y = 20.0 + ((2+.4)*sin(glfwGetTime()+1.5));
        texts[4].y = 20.0 + ((2+.4)*sin(glfwGetTime()+1.5));

    }

    texts[0].y = 50.0 + ((2+.4)*sin(glfwGetTime()+1.5));
    texts[1].y = 50.0 + ((2+.18)*sin(glfwGetTime()+2.4));
    texts[2].y = 50.0 + ((2)*sin(glfwGetTime()+4.3));
    float voff = vectorstringoffset(texts[0].text);
    texts[0].x = 50.0 + voff + (2*cos(1.5*glfwGetTime()+1.5));
    texts[1].x = 50.0 + voff + (2*cos(1.5*glfwGetTime()+2.4));
    texts[2].x = 50.0 + voff + (2*cos(1.5*glfwGetTime()+4.3));
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
    float t = glfwGetTime();
    glm::mat3x3 m;
    m[0][0] = cos(t/3);
    m[1][0] = -sin(t/3);
    m[0][1] = sin(t/3);
    m[1][1] = cos(t/3);
    glm::mat3x3 m2;
    m2[0][0] = cos((t+0.5)/2);
    m2[1][0] = -sin((t+0.5)/2);
    m2[0][1] = sin((t+0.5)/2);
    m2[1][1] = cos((t+0.5)/2);
    auto f = [](glm::vec3 loc) {
        if(loc.x < 0) {
            loc.x = (-loc.x) + 50;
        }
        if(loc.y < 0) {
            loc.y = (-loc.y) + 50;
        }
        //int x = (int)(loc.x/50) + (int)(loc.y/50);
        uint16_t xloc  = ((int)loc.x % 153);
        uint16_t yloc  = ((int)loc.y % 153);

        glm::vec3 color(bmp.data[yloc*153*3 + xloc*3 + 0], bmp.data[yloc*153*3 + xloc*3 + 1], bmp.data[yloc*153*3 + xloc*3 + 2]);
        return color;
//        if(x % 2) {
//            return glm::vec3(83, 212, 230);
//        }
//        return glm::vec3(255, 192, 203);
    };
    glm::vec3 c(0, 0, 0);
    c.x += cos(t)*200;
    c.y += sin(t)*200;
    c.x *= sin(t)+1;
    c.y *= sin(t)+1;
    c = m * c;
        for(size_t x = 0; x < 800; x++) {
            for(size_t y = 0; y < 450; y++) {
                glm::vec3 l((int)x - 400, (int)y - 225 , 0);
                if(y % 2 == 0 && (t > 8.0)) {
                    l.x += cos(t)*200;
                    l.y += sin(t)*200;
                    l.x *= sin(t)+1;
                    l.y *= sin(t)+1;
                    //l = m * l;
                } else {
                    l.x += cos(t+0.5)*200;
                    l.y += sin(t+0.5)*200;
                    l.x *= sin(t+0.5)+1;
                    l.y *= sin(t+0.5)+1;
                    //l = m2 * l;
                }
                float xdif = l.x-c.x;
                float ydif = l.y-c.y;
                float m = std::fmin(1.0f,(1.0f/(sqrtf(xdif*xdif + ydif*ydif)/256.0f)));
                glm::vec3 color = f(l)*(float)std::fmin(1.0f, glfwGetTime()/4.0);
                fb.fb[y*800*3 + x*3] = color.r;
                fb.fb[y*800*3 + x*3 + 1] = color.g;
                fb.fb[y*800*3 + x*3 + 2] = color.b;
            }
        }
    //cudaGetLastError();
    //    float t = glfwGetTime();
    //#pragma omp parallel for
    //    for(size_t x = 0; x < 1600; x++) {
    //        for(size_t y = 0; y < 900; y++) {
    //            uint8_t amp = (sine_wave[(x/4 + (int)(t*200)) & 0xff] + sine_wave[(y/4 + (int)(glfwGetTime()*50)) & 0xff] + (int)(glfwGetTime()*10)) + sine_wave[(x/4+y/4) & 0xff] + sine_wave[(int)(x/4+134*(t))&0xff];
    //            fb.fb[y*1600*3 + x*3] = ((uint16_t)amp * 16) >> 8;
    //            fb.fb[y*1600*3 + x*3 + 1] = ((uint16_t)amp * 32) >> 8;
    //            fb.fb[y*1600*3 + x*3 + 2] = ((uint16_t)amp * 64) >> 8;

    //        }
    //    }
//    memset(fb.fb, 255, fb.x*fb.y*3);
//    for( float y = 1.3 ; y >= -1.1 ; y -= 0.0075 ){
//        for( float x = -1.2 ; x <= 1.2 ; x += 0.00625 ) {
//            uint32_t xa = 208 + (x + 1.2)/0.00625;
//            uint32_t ya = 96 +  ((y+1.1) / 0.0075);
//            if( pow((x*x+y*y-1.0),3) - x*x*y*y*y <= 0.0 ) {
//                //                float amp  = 1.0;
//                //                float tadj = t-4;
//                //                if(t > 6.0) {
//                //                    amp = (cos(2*sqrt(((x/4)*(x/4)) + (y/4*(y/4))) + t-6.0) + 1.0)/2.0;
//                //                }
//                //                if(t > 8.9) {
//                //                    amp = 0;
//                //                }
//                //                if(((rand()) % 5  != 0) || (t < 6)) {
//                //                    fb.fb[ya*800*3 + xa*3] = 255 * amp;
//                //                    fb.fb[ya*800*3 + xa*3  + 1] = 0;
//                //                    fb.fb[ya*800*3 + xa*3  + 2] = 138 * amp;
//                //                } else if(t > 6) {
//                //                    fb.fb[ya*800*3 + xa*3] = 255 * amp;
//                //                    fb.fb[ya*800*3 + xa*3  + 1] = 255 * amp;
//                //                    fb.fb[ya*800*3 + xa*3  + 2] = 255 * amp;
//                //                }
//                float amp = ((fmin(fmax(t, 8.0),10.0)-8.0)/2.0);
//                fb.fb[ya*800*3 + xa*3] = 255 * amp;
//                fb.fb[ya*800*3 + xa*3 + 1] = 255 * amp;
//                fb.fb[ya*800*3 + xa*3 + 2] = 255 * amp;
//            }
//        }
//    }
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
                float yoff = ((rand() % 255) - 127.5)/510.0;
                float xoff = ((rand() % 255) - 127.5)/510.0;
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
    window = glfwCreateWindow(800, 450, "t", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();
    Scene man(currentbackend, 4, PrepFrameTest ,DrawFrameTest);
    Framebuffer fb;
    fb.x = 800;
    fb.y = 450;
    fb.fb = (uint8_t*)malloc(fb.x*fb.y*3);

    //cudaDeviceSynchronize();
    man.render(fb);
}
