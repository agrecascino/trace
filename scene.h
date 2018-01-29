#ifndef SCENE_H
#define SCENE_H
#include "structs.h"
#include <embree3/rtcore.h>
#include "ctpl.h"
#include <unordered_map>
#include <CL/cl.h>


class Scene {
public:
    Scene(RenderBackend backend, size_t nthreads);

    OwnedHandle AddObject(Intersectable* obj);

    OwnedHandle AddObject(std::vector<Intersectable*> objects);

    void RemoveObjectsByHandle(OwnedHandle handle);

    void RegenerateObjectCache();

    void AddLight(Light* obj);

    void SetCameraConfig(CameraConfig &cfg);

    glm::vec3 GenerateNoiseVector();

    void tapecast(Ray *r, Intersection *hit, unsigned int numrays);

    void embreecast(RTCRayHit16 *rhit, Intersection *hit, int *valid) ;

    void inline GenerateScreenVectors(RTCRayHit16 *r, glm::vec3 correctedright, glm::vec3 localup, size_t xf, size_t yf, size_t xl, size_t yl, int numrays, int *valid);

    void RenderSliceTape(size_t yfirst, size_t ylast, Framebuffer &fb);

    void RenderSliceEmbree(size_t yfirst, size_t ylast, Framebuffer &fb);

    void SwitchBackend(RenderBackend back);
    void render(Framebuffer &fb);

    inline void fast_srand(int seed);

    // Compute a pseudorandom integer.
    // Output value in range [0, 32767]
    inline int fast_rand(void);

    ~Scene();
private:
    SphereCL *spheres_buf = NULL;
    TriangleCL *triangles_buf = NULL;
    LightCL *lights_buf = NULL;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem buffer, cl_tris, cl_spheres, cl_lights;
    RenderBackend backend;
    RTCGeometry *geometry = NULL;
    RTCDevice device = rtcNewDevice("verbose=1");
    RTCScene scene;
    unsigned int g_seed;
    bool frun = true;
    std::atomic<size_t> current_id;
    CameraConfig config;
    std::unordered_map<size_t, Intersectable*> intersectables;
    unsigned long tricount = 0;
    unsigned long spherecount = 0;
    std::vector<Light*> lights;
    ctpl::thread_pool pool;
};


#endif // SCENE_H
