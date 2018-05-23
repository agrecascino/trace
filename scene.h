#ifndef SCENE_H
#define SCENE_H
#include "structs.h"
#include <embree3/rtcore.h>
#include "ctpl.h"
#include <unordered_map>
#include <CL/cl.h>
#include <functional>

class Scene {
public:
    Scene(RenderBackend backend, size_t nthreads,
          std::function<int(Scene*, Framebuffer&)> prep,
          std::function<void(Scene*, Framebuffer&)> draw);

    OwnedHandle AddObject(Intersectable* obj);

    OwnedHandle AddObject(std::vector<Intersectable*> objects);

    void RemoveObjectsByHandle(OwnedHandle handle);

    void RegenerateObjectCache();

    void TranslateAndRotate();

    void GPUUploadGeometry();

    void GPUUploadMaterials();

    void AddLight(Light* obj);

    void SetCameraConfig(CameraConfig &cfg);

    glm::vec3 GenerateNoiseVector();

    void shade(Intersection &hit, glm::vec3 &fcolor, Ray &ro);

    void tapecast(Ray *r, Intersection *hit, unsigned int numrays);

    void embreecast(RTCRayHit16 *rhit, Intersection *hit, int *valid) ;

    void inline GenerateScreenVectors(RTCRayHit16 *r, glm::vec3 correctedright, glm::vec3 localup, size_t xf, size_t yf, size_t xl, size_t yl, int numrays, int *valid);

    void RenderSliceTape(size_t yfirst, size_t ylast, Framebuffer &fb);

    void RenderSliceEmbree(size_t yfirst, size_t ylast, Framebuffer &fb);

    void SwitchBackend(RenderBackend back);
    void render(Framebuffer &fb);

    void fast_srand(int seed);

    // Compute a pseudorandom integer.
    // Output value in range [0, 32767]
    uint64_t fast_rand(void);

    ~Scene();
private:
    RenderBackend backend;
    std::function<int(Scene*, Framebuffer&)> prepframe;
    std::function<void(Scene*, Framebuffer&)> drawframe;
    SphereCL *spheres_buf = NULL;
    TriangleCL *triangles_buf = NULL;
    LightCL *lights_buf = NULL;
    AreaLightCL *alights_buf = NULL;
    cl_uint *emittersets_buf = NULL;
    cl_float *halton_buf = NULL;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem buffer = NULL, cl_tris = NULL, cl_spheres = NULL, cl_lights = NULL, cl_alights = NULL, cl_emittersets = NULL, cl_halton = NULL;
    RTCGeometry *geometry = NULL;
    RTCDevice device = rtcNewDevice("verbose=1");
    RTCScene scene;
    unsigned int g_seed;
    std::atomic<size_t> current_id;
    CameraConfig config;
    std::unordered_map<size_t, Intersectable*> intersectables;
    unsigned long tricount = 0;
    unsigned long spherecount = 0;
    unsigned long alightcount = 0;
    unsigned long emittersetscount = 0;
    std::vector<Light*> lights;
    std::unordered_map<uint64_t, AreaLight> alightcache;
    ctpl::thread_pool pool;
};


#endif // SCENE_H
