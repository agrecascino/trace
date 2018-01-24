#include "scene.h"
#include "triangle.h"
#include "sphere.h"
#include <fstream>

Scene::Scene(RenderBackend backend, size_t nthreads) : pool(nthreads), backend(backend) {
    if(backend == Embree)
        scene = rtcNewScene(device);
    if(backend == OpenCL) {
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);
        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
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
        if(result)
            throw RendererInitializationException();
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
                //RTCBoundsFunction

            }
        }
        rtcCommitScene(scene);
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

void Scene::embreecast(RTCRayHit *rhit, Intersection *hit, unsigned int numrays) {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    rtcIntersect1M(scene, &context, rhit, numrays, sizeof(RTCRayHit));
    for(unsigned int i = 0; i < numrays; i++) {
        if(rhit[i].ray.tfar < 65535.0f) {
            RTCGeometry geo = rtcGetGeometry(scene, rhit[i].hit.geomID);
            Intersectable *a = (Intersectable*)rtcGetGeometryUserData(geo);
            hit[i].t = rhit[i].ray.tfar;
            hit[i].mat = a->getMaterial();
            Ray r;
            r.direction = glm::vec3(rhit[i].ray.dir_x, rhit[i].ray.dir_y, rhit[i].ray.dir_z);
            r.origin = glm::vec3(rhit[i].ray.org_x, rhit[i].ray.org_y, rhit[i].ray.org_z);
            hit[i].normal = a->getNormal(r, rhit[i].ray.tfar);
            hit[i].point = r.origin + rhit[i].ray.tfar*r.direction;
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
    for(size_t x = 0; x < fb.x; x++) {
        for(size_t y = yfirst; y < ylast; ) {
            uint32_t np = ylast-y;
            Ray r[(np > 7) ? 8 : 1];
            for(int i = 0; i < ((np > 7) ? 8: 1); i++) {
                float nx = ((float)x / fb.x) - 0.5;
                float ny = ((float)(y+i) / fb.y) - 0.5;
                r[i].origin = config.center;
                r[i].direction = (correctedright * nx) + (-localup * ny) + config.lookat;
                r[i].direction = glm::normalize(r[i].direction);
            }
            Intersection hit[(np > 7) ? 8 : 1];
            tapecast(r, hit, (np > 7) ? 8 : 1);
            for(int i = 0; i < ((np > 7) ? 8 : 1); i++) {
                if(!hit[i].intersected) {
                    fb.fb[((fb.x * (y+i)) + x)*3] = 100;
                    fb.fb[((fb.x * (y+i)) + x)*3 + 1] = 149;
                    fb.fb[((fb.x * (y+i)) + x)*3 + 2] = 237;
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
                fb.fb[((fb.x * (y+i)) + x)*3] = fminf(fmaxf(0,fcolor[i].x*255),255);
                fb.fb[((fb.x * (y+i)) + x)*3 + 1] = fminf(fmaxf(0,fcolor[i].y*255),255);
                fb.fb[((fb.x * (y+i)) + x)*3 + 2] = fminf(fmaxf(0,fcolor[i].z*255),255);
            }
            y += (np > 7) ? 8 : 1;
        }
    }
}

void Scene::RenderSliceEmbree(size_t yfirst, size_t ylast, Framebuffer &fb) {
    glm::vec3 camright = glm::cross(config.up,config.lookat);
    glm::vec3 localup = glm::cross(camright, config.lookat);
    float aspect = fb.x/(float)fb.y;
    glm::vec3 correctedright = aspect * camright;
    for(size_t x = 0; x < fb.x; x++) {
        for(size_t y = yfirst; y < ylast; ) {
            uint32_t np = ylast-y;
            RTCRayHit r[(np > 7) ? 8 : 1];
            for(int i = 0; i < ((np > 7) ? 8: 1); i++) {
                float nx = ((float)x / fb.x) - 0.5;
                float ny = ((float)(y+i) / fb.y) - 0.5;
                glm::vec3 origin = config.center;
                glm::vec3 direction = (correctedright * nx) + (-localup * ny) + config.lookat;
                direction = glm::normalize(direction);
                r[i].ray.dir_x = direction.x;
                r[i].ray.dir_y = direction.y;
                r[i].ray.dir_z = direction.z;
                r[i].ray.org_x = origin.x;
                r[i].ray.org_y = origin.y;
                r[i].ray.org_z = origin.z;
                r[i].ray.time = 0.0;
                r[i].ray.tnear = 0.0f;
                r[i].ray.tfar = 65535.0f;
            }
            Intersection hit[(np > 7) ? 8 : 1];
            embreecast(r, hit, (np > 7) ? 8 : 1);
            for(int i = 0; i < ((np > 7) ? 8 : 1); i++) {
                if(!hit[i].intersected) {
                    fb.fb[((fb.x * (y+i)) + x)*3] = 100;
                    fb.fb[((fb.x * (y+i)) + x)*3 + 1] = 149;
                    fb.fb[((fb.x * (y+i)) + x)*3 + 2] = 237;
                }
            }
            glm::vec3 fcolor[(np > 7) ? 8 : 1];
            for(int i = 0; i < ((np > 7) ? 8 : 1); i++) {
                if(!hit[i].intersected)
                    continue;
                for(Light *light : lights) {
                    glm::vec3 l = light->location - hit[i].point;
                    glm::vec3 n = hit[i].normal;
                    glm::vec3 rdirect = glm::vec3(r[i].ray.dir_x, r[i].ray.dir_y, r[i].ray.dir_z);
                    if(glm::dot(n, -rdirect) < 0) {
                        n = -n;
                    }
                    float dt = glm::dot(glm::normalize(l), n);
                    RTCRayHit s;
                    glm::vec3 origin = glm::vec3(hit[i].point) + (n * 0.001f);
                    glm::vec3 direction = glm::normalize((light->location - hit[i].point));
                    s.ray.dir_x = direction.x;
                    s.ray.dir_y = direction.y;
                    s.ray.dir_z = direction.z;
                    s.ray.org_x = origin.x;
                    s.ray.org_y = origin.y;
                    s.ray.org_z = origin.z;
                    s.ray.tnear = 0.0f;
                    s.ray.tfar = 65535.0f;
                    Intersection s_hit;
                    embreecast(&s, &s_hit, 1);
                    if((!s_hit.intersected) || (glm::distance(s_hit.point, origin) > glm::distance(origin, light->location))) {
                        fcolor[i] += ((light->color*dt)* hit[i].mat.color);
                    }
                }
            }

            for(int i = 0; i < ((np > 7) ? 8 : 1); i++) {
                if(!hit[i].intersected)
                    continue;
                fb.fb[((fb.x * (y+i)) + x)*3] = fminf(fmaxf(0,fcolor[i].x*255),255);
                fb.fb[((fb.x * (y+i)) + x)*3 + 1] = fminf(fmaxf(0,fcolor[i].y*255),255);
                fb.fb[((fb.x * (y+i)) + x)*3 + 2] = fminf(fmaxf(0,fcolor[i].z*255),255);
            }
            y += (np > 7) ? 8 : 1;
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

