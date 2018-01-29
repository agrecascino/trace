#include "scene.h"
#include "triangle.h"
#include "sphere.h"
#include <fstream>
#include <string.h>
static glm::vec3 lerp(const glm::vec3 &v1, const glm::vec3 &v2, const float &r) {
    return v1 + (v2 - v1) * r;
}

static glm::vec3 quadlerp(const glm::vec3 &v1,
                          const glm::vec3 &v2,
                          const glm::vec3 &v3,
                          const glm::vec3 &v4,
                          float u,
                          float v) {
    return lerp(lerp(v1, v2, u), lerp(v3, v4, u), v);
}

Scene::Scene(RenderBackend backend, size_t nthreads) : pool(nthreads), backend(backend) {
    if(backend == Embree)
        scene = rtcNewScene(device);
    if(backend == OpenCL) {
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);
        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device, 0, NULL);
        std::ifstream file("clcast.cl");
        std::string source;
        std::string line;
        while(std::getline(file, line)) {
            source += line;
        }
        cl_ulong maxSize;
        clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxSize, 0);
        const char* str = source.c_str();
        cl_program program = clCreateProgramWithSource(context, 1, &str, NULL, NULL);
        cl_int result = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if(result) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            std::string s;
            s.resize(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, (char*)s.c_str(), NULL);

            throw RendererInitializationException(s);
        }
        kernel = clCreateKernel(program, "main", NULL);
    }
    current_id = 0;
    fast_srand(0);
}

OwnedHandle Scene::AddObject(Intersectable* obj) {
    intersectables[current_id] = obj;
    OwnedHandle h(std::vector<size_t>(1, current_id));
    current_id++;
    RegenerateObjectCache();
    return h;
}

OwnedHandle Scene::AddObject(std::vector<Intersectable*> objects) {
    std::vector<size_t> handles;
    for(Intersectable* obj : objects) {
        handles.push_back(current_id);
        intersectables[current_id] = obj;
        current_id++;
    }
    RegenerateObjectCache();
    return OwnedHandle(handles);
}

void Scene::RemoveObjectsByHandle(OwnedHandle handle) {
    for(size_t i : handle.identifiers_owned) {
        intersectables.erase(i);
    }
    RegenerateObjectCache();
}

void Scene::SwitchBackend(RenderBackend back) {
    backend = back;
    RegenerateObjectCache();
}


void Scene::RegenerateObjectCache() {
    if(backend == Embree) {
        rtcReleaseScene(scene);
        scene = rtcNewScene(device);
        geometry = new RTCGeometry[intersectables.size()];
        for(unsigned long i = 0; i < intersectables.size(); i++) {
            if(typeid(*intersectables[i]) == typeid(Triangle)) {
                geometry[i] = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
                float *buffer = (float*)rtcSetNewGeometryBuffer(geometry[i], RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(glm::vec3), 3);
                float *vert = ((Triangle*)intersectables[i])->getVertexBuffer();
                for(unsigned long z = 0; z < 3*3; z++) {
                    buffer[z] = vert[z];
                }
                uint32_t *indices = (uint32_t*)rtcSetNewGeometryBuffer(geometry[i], RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(uint32_t)*3, 3);
                for(unsigned long z = 0; z < 3*3; z++) {
                    indices[z] = z;
                }
                rtcSetGeometryUserData(geometry[i], intersectables[i]);
                rtcCommitGeometry(geometry[i]);
                rtcAttachGeometry(scene, geometry[i]);
                rtcReleaseGeometry(geometry[i]);
            } else if(typeid(*intersectables[i]) == typeid(Sphere)) {
                geometry[i] = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER);
                rtcSetGeometryUserPrimitiveCount(geometry[i], 1);
                rtcSetGeometryUserData(geometry[i], intersectables[i]);
                rtcSetGeometryBoundsFunction(geometry[i], sphereBoundsFunc, nullptr);
                rtcSetGeometryOccludedFunction(geometry[i], sphereOccludedFunc);
                rtcSetGeometryIntersectFunction(geometry[i], sphereIntersectFunc);
                rtcCommitGeometry(geometry[i]);
                ((Sphere*)intersectables[i])->setGeomID(rtcAttachGeometry(scene, geometry[i]));
                rtcReleaseGeometry(geometry[i]);
            }
        }
        delete[] geometry;
        rtcCommitScene(scene);
    } else if(backend == OpenCL) {
        spherecount = 0;
        tricount = 0;
        for(unsigned long i = 0; i < intersectables.size(); i++) {
            if(typeid(*intersectables[i]) == typeid(Triangle)) {
                tricount++;
            } else if(typeid(*intersectables[i]) == typeid(Sphere)) {
                spherecount++;
            }
        }

        triangles_buf = new TriangleCL[tricount];
        spheres_buf = new SphereCL[spherecount];
        int tc = 0;
        int sc = 0;
        for(unsigned long i = 0; i < intersectables.size(); i++) {
            if(typeid(*intersectables[i]) == typeid(Triangle)) {
                memcpy(triangles_buf[tc].pts, ((Triangle*)intersectables[i])->getVertexBuffer(), 3*sizeof(glm::vec3));
                triangles_buf[tc].mat = intersectables[i]->getMaterial();
                tc++;
            } else if(typeid(*intersectables[i]) == typeid(Sphere)) {
                spheres_buf[sc].origin = ((Sphere*)intersectables[i])->origin;
                spheres_buf[sc].radius = ((Sphere*)intersectables[i])->radius;
                spheres_buf[sc].mat = intersectables[i]->getMaterial();
                sc++;
            }
        }
        cl_tris = clCreateBuffer(context, CL_MEM_READ_ONLY, tricount * sizeof(TriangleCL), NULL, 0);
        cl_lights = clCreateBuffer(context, CL_MEM_READ_ONLY, lights.size() * sizeof(LightCL), NULL, 0);
        cl_spheres = clCreateBuffer(context, CL_MEM_READ_ONLY, spherecount * sizeof(TriangleCL), NULL, 0);

    }
}

void Scene::AddLight(Light* obj) {
    lights.push_back(obj);
}

void Scene::SetCameraConfig(CameraConfig &cfg) {
    config = cfg;
}

glm::vec3 Scene::GenerateNoiseVector() {
    int x = fast_rand() % UINT16_MAX;
    int y = fast_rand() % UINT16_MAX;
    int z = fast_rand() % UINT16_MAX;
    return glm::vec3(x, y, z);
}
void Scene::tapecast(Ray *r, Intersection *hit, unsigned int numrays) {
#pragma omp simd
    for(unsigned int i = 0; i < numrays; i++) {
        for(auto obj : intersectables) {
            Intersection h = obj.second->intersect(r[i]);
            hit[i] = (h.intersected && (h.t < hit[i].t)) ? h : hit[i];
        }
    }
}

void Scene::embreecast(RTCRayHit16 *rhit, Intersection *hit, int *valid) {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    rtcIntersect16(valid, scene, &context, rhit);
    //rtcIntersect1M(scene, &context, rhit, numrays, sizeof(RTCRayHit));
    for(unsigned int i = 0; i < 16; i++) {
        if(valid[i] != -1)
            continue;
        if(rhit->ray.tfar[i] < 65535.0f) {
            RTCGeometry geo = rtcGetGeometry(scene, rhit->hit.geomID[i]);
            Intersectable *a = (Intersectable*)rtcGetGeometryUserData(geo);
            hit[i].t = rhit->ray.tfar[i];
            hit[i].mat = a->getMaterial();
            Ray r;
            r.direction = glm::vec3(rhit->ray.dir_x[i], rhit->ray.dir_y[i], rhit->ray.dir_z[i]);
            r.origin = glm::vec3(rhit->ray.org_x[i], rhit->ray.org_y[i], rhit->ray.org_z[i]);
            hit[i].normal = a->getNormal(r, rhit->ray.tfar[i]);
            hit[i].point = r.origin + rhit->ray.tfar[i]*r.direction;
            hit[i].intersected = true;
        } else {
            hit[i].intersected = false;
        }
    }
}

void Scene::RenderSliceTape(size_t yfirst, size_t ylast, Framebuffer &fb) {
    glm::vec3 camright = glm::cross(config.up,config.lookat);
    glm::vec3 localup = glm::cross(camright, config.lookat);
    float aspect = fb.x/(float)fb.y;
    glm::vec3 correctedright = aspect * camright;
    for(size_t y = yfirst; y < ylast; y++) {
        for(size_t x = 0; x < fb.x;) {
            uint32_t np = fb.x-x;
            Ray r[(np > 7) ? 8 : 1];
#pragma omp simd
            for(int i = 0; i < ((np > 7) ? 8: 1); i++) {
                float nx = ((float)(x+i) / fb.x) - 0.5;
                float ny = ((float)y / fb.y) - 0.5;
                r[i].origin = config.center;
                r[i].direction = (correctedright * nx) + (-localup * ny) + config.lookat;
                r[i].direction = glm::normalize(r[i].direction);
            }
            Intersection hit[(np > 7) ? 8 : 1];
            tapecast(r, hit, (np > 7) ? 8 : 1);
            for(int i = 0; i < ((np > 7) ? 8 : 1); i++) {
                if(!hit[i].intersected) {
                    fb.fb[((fb.x * (y)) + (x+i))*3] = 100;
                    fb.fb[((fb.x * (y)) + (x+i))*3 + 1] = 149;
                    fb.fb[((fb.x * (y)) + (x+i))*3 + 2] = 237;
                }
            }
            glm::vec3 fcolor[(np > 7) ? 8 : 1];
            for(int i = 0; i < ((np > 7) ? 8 : 1); i++) {
                if(!hit[i].intersected)
                    continue;
                for(Light *light : lights) {
                    glm::vec3 l = light->location - hit[i].point;
                    glm::vec3 n = hit[i].normal;
                    if(glm::dot(n, -r[i].direction) < 0) {
                        n = -n;
                    }
                    float dt = glm::dot(glm::normalize(l), n);
                    Ray s;
                    s.origin = glm::vec3(hit[i].point) + (n * 0.001f);
                    s.direction = glm::normalize((light->location - hit[i].point));
                    Intersection s_hit;
                    tapecast(&s, &s_hit, 1);
                    if((!s_hit.intersected) || (glm::distance(s_hit.point, s.origin) > glm::distance(s.origin, light->location))) {
                        fcolor[i] += (((light->color*dt))* hit[i].mat.color);
                    }
                }
            }

            for(int i = 0; i < ((np > 7) ? 8 : 1); i++) {
                if(!hit[i].intersected)
                    continue;
                fb.fb[((fb.x * (y)) + (x+i))*3] = fminf(fmaxf(0,fcolor[i].x*255),255);
                fb.fb[((fb.x * (y)) + (x+i))*3 + 1] = fminf(fmaxf(0,fcolor[i].y*255),255);
                fb.fb[((fb.x * (y)) + (x+i))*3 + 2] = fminf(fmaxf(0,fcolor[i].z*255),255);
            }
            x += (np > 7) ? 8 : 1;
        }
    }
}

void inline Scene::GenerateScreenVectors(RTCRayHit16 *r, glm::vec3 correctedright, glm::vec3 localup, size_t xf, size_t yf, size_t xl, size_t yl, int numrays, int *valid) {
    for(int i = 0; i < numrays; i++) {
        float nx = ((float)(xf+i) / xl) - 0.5;
        float ny = ((float)yf / yl) - 0.5;
        glm::vec3 origin = config.center;
        glm::vec3 direction = (correctedright * nx) + (-localup * ny) + config.lookat;
        direction = glm::normalize(direction);
        r->ray.dir_x[i] = direction.x;
        r->ray.dir_y[i] = direction.y;
        r->ray.dir_z[i] = direction.z;
        r->ray.org_x[i] = origin.x;
        r->ray.org_y[i] = origin.y;
        r->ray.org_z[i] = origin.z;
        r->ray.time[i] = 0.0;
        r->ray.tnear[i] = 0.0f;
        r->ray.tfar[i] = 65535.0f;
        valid[i] = -1;
    }
}

void Scene::RenderSliceEmbree(size_t yfirst, size_t ylast, Framebuffer &fb) {
    glm::vec3 camright = glm::cross(config.up,config.lookat);
    glm::vec3 localup = glm::cross(camright, config.lookat);
    float aspect = fb.x/(float)fb.y;
    glm::vec3 correctedright = aspect * camright;
    for(size_t y = yfirst; y < ylast; y++ ) {
        for(size_t x = 0; x < fb.x; ) {
            uint32_t np = std::min((size_t)16, fb.x-x);
            RTCRayHit16 r;
            int valid[16];
            memset(valid, 0, 16);
            GenerateScreenVectors(&r, correctedright, localup, x, y, fb.x, fb.y, np, valid);
            Intersection hit[np];
            embreecast(&r, hit, valid);
            for(size_t i = 0; i < np; i++) {
                if(!hit[i].intersected) {
                    fb.fb[((fb.x * (y)) + (x+i))*3] = 100;
                    fb.fb[((fb.x * (y)) + (x+i))*3 + 1] = 149;
                    fb.fb[((fb.x * (y)) + (x+i))*3 + 2] = 237;
                }
            }
            glm::vec3 fcolor[np];
            for(size_t i = 0; i < np; i++) {
                if(!hit[i].intersected)
                    continue;
                for(Light *light : lights) {
                    glm::vec3 l = light->location - hit[i].point;
                    glm::vec3 n = hit[i].normal;
                    glm::vec3 rdirect = glm::vec3(r.ray.dir_x[i], r.ray.dir_y[i], r.ray.dir_z[i]);
                    if(glm::dot(n, -rdirect) < 0) {
                        n = -n;
                    }
                    float dt = glm::dot(glm::normalize(l), n);
                    RTCRayHit16 s;
                    int valid[16];
                    memset(valid, 0, 16);
                    valid[0] = -1;
                    glm::vec3 origin = glm::vec3(hit[i].point) + (n * 0.001f);
                    glm::vec3 direction = glm::normalize((light->location - hit[i].point));
                    //                    s.ray.dir_x[0] = direction.x;
                    //                    s.ray.dir_y[0] = direction.y;
                    //                    s.ray.dir_z[0] = direction.z;
                    //                    s.ray.org_x[0] = origin.x;
                    //                    s.ray.org_y[0] = origin.y;
                    //                    s.ray.org_z[0] = origin.z;
                    //                    s.ray.tnear[0] = 0.0f;
                    //                    s.ray.tfar[0] = 65535.0f;
                    Ray r;
                    r.origin = origin;
                    r.direction = direction;
                    Intersection s_hit;
                    //embreecast(&s, &s_hit, valid);
                    tapecast(&r, &s_hit, 1);
                    if((!s_hit.intersected) || (glm::distance(s_hit.point, origin) > glm::distance(origin, light->location))) {
                        fcolor[i] += ((light->color*dt)* hit[i].mat.color);
                    }
                }
            }

            for(size_t i = 0; i < np; i++) {
                if(!hit[i].intersected)
                    continue;
                fb.fb[((fb.x * (y)) + (x+i))*3] = fminf(fmaxf(0,fcolor[i].x*255),255);
                fb.fb[((fb.x * (y)) + (x+i))*3 + 1] = fminf(fmaxf(0,fcolor[i].y*255),255);
                fb.fb[((fb.x * (y)) + (x+i))*3 + 2] = fminf(fmaxf(0,fcolor[i].z*255),255);
            }
            x += np;
        }
    }
}

void Scene::render(Framebuffer &fb) {
    if(!(fb.x) || !(fb.y)) {
        return;
    }
    size_t ystep = fb.y/8;
    std::vector<std::future<void>> f(8);
#pragma omp parallel for
    for(size_t i = 0; i < 8; i++) {
        switch(backend) {
        case Rendertape:
            RenderSliceTape(ystep*i, ystep*(i+1), fb);
            break;
        case Embree:
            RenderSliceEmbree(ystep*i, ystep*(i+1), fb);
            break;
        case OpenCL:
            buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, fb.x*fb.y*sizeof(cl_uchar3), NULL, 0);
            clSetKernelArg(kernel, 0, sizeof(buffer), (void*)&buffer);
            clSetKernelArg(kernel, 1, sizeof(cl_uint),(void*)&fb.x);
            clSetKernelArg(kernel, 2, sizeof(cl_uint),(void*)&fb.y);
            clSetKernelArg(kernel, )
        }
    }
}

inline void Scene::fast_srand(int seed) {
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int Scene::fast_rand(void) {
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

Scene::~Scene() {
    rtcReleaseDevice(device);
}

