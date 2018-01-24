#include "sphere.h"

Sphere::Sphere(glm::vec3 origin, float radius, glm::vec3 color) : origin(origin), radius(radius), color(color) {}
glm::vec3 Sphere::getNormal(Ray &ray, float &t) { return ((ray.origin + (t*ray.direction) - origin)) / radius; }
Intersection Sphere::intersect(Ray &ray) {
    Intersection ret;
    glm::vec3 p = ray.origin - origin;
    float rpow2 = radius*radius;
    float p_d = glm::dot(p, ray.direction);

    if(p_d > 0 || dot(p, p) < rpow2)
        return ret;

    glm::vec3 a = p - p_d * ray.direction;
    float apow2 = glm::dot(a, a);
    if(apow2 > rpow2)
        return ret;
    ret.intersected = true;
    float h = sqrtf(rpow2 - apow2);
    glm::vec3 i = a - h*ray.direction;
    ret.point = origin + i;
    ret.t = ((ret.point - ray.origin) / ray.direction).x;
    ret.normal = i/radius;
    //ret.normal = getNormal(ray, ret.t);
    Material mat;
    mat.color = color;
    ret.mat = mat;
    return ret;
}

IntersectableInternalOptimized* Sphere::optimize() {
    SphereInternalOptimized *o = new SphereInternalOptimized;
    Material mat;
    mat.color = color;
    mat.reflective = false;
    o->mat = mat;
    o->type = SphereType;
    o->origin = origin;
    o->radius = radius;
    return (IntersectableInternalOptimized*)o;
}

void sphereBoundsFunc(const struct RTCBoundsFunctionArguments* args) {
    const Sphere* sphere = (const Sphere*)args->geometryUserPtr;
    RTCBounds* bounds_o = args->bounds_o;
    bounds_o->lower_x = sphere->origin.x - sphere->radius;
    bounds_o->lower_y = sphere->origin.y - sphere->radius;
    bounds_o->lower_z = sphere->origin.z - sphere->radius;
    bounds_o->upper_x = sphere->origin.x + sphere->radius;
    bounds_o->upper_y = sphere->origin.y + sphere->radius;
    bounds_o->upper_z = sphere->origin.z + sphere->radius;
}

void sphereOccludedFunc(const RTCOccludedFunctionNArguments* args) {
    const int* valid = args->valid;
    void* ptr  = args->geometryUserPtr;
    RTCRayHitN* rays = (RTCRayHitN*)args->ray;
    RTCRay *ray = (RTCRay*)rays;
    glm::vec3 origin = glm::vec3(ray->org_x, ray->org_y, ray->org_z);
    glm::vec3 direction = glm::vec3(ray->dir_x, ray->dir_y, ray->dir_z);
    assert(args->N == 1);
     if (!valid[0])
        return;
    const Sphere *sphere = (Sphere*)ptr;
    glm::vec3 p = origin - sphere->origin;
    float rpow2 = sphere->radius*sphere->radius;
    float p_d = glm::dot(p, direction);
    if(p_d > 0 || dot(p, p) < rpow2)
        return;
    glm::vec3 a = p - p_d * direction;
    float apow2 = glm::dot(a, a);
    if(apow2 > rpow2)
        return;
    float h = sqrtf(rpow2 - apow2);
    glm::vec3 i = a - h*direction;
    glm::vec3 point = origin + i;
    float t =((point - origin) / direction).x;
    ray->tfar = ((ray->tfar > t) & (ray->tnear < t)) ? -INFINITY : ray->tfar;
}


void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args) {
    if (args->context == nullptr)
        return;
    const int* valid = args->valid;
    RTCRayHitN* rayhits = (RTCRayHitN*)args->rayhit;
    unsigned int n = args->N;
    RTCRayN* rays = RTCRayHitN_RayN(rayhits, n);
    RTCHitN* hits = RTCRayHitN_HitN(rayhits, n);
    for(size_t i = 0; i < n; i++) {
        if (valid[i] != -1) continue;
        glm::vec3 origin = glm::vec3(RTCRayN_org_x(rays, n, i), RTCRayN_org_y(rays, n, i), RTCRayN_org_z(rays, n, i));
        glm::vec3 direction = glm::vec3(RTCRayN_dir_x(rays, n, i), RTCRayN_dir_y(rays, n, i), RTCRayN_dir_z(rays, n, i));
        void* ptr  = args->geometryUserPtr;
        const Sphere *sphere = (Sphere*)ptr;
        glm::vec3 p = origin - sphere->origin;
        float rpow2 = sphere->radius*sphere->radius;
        float p_d = glm::dot(p, direction);
        if(p_d > 0 || dot(p, p) < rpow2)
            return;
        glm::vec3 a = p - p_d * direction;
        float apow2 = glm::dot(a, a);
        if(apow2 > rpow2)
            return;
        float h = sqrtf(rpow2 - apow2);
        glm::vec3 i_vec = a - h*direction;
        glm::vec3 point = origin + i_vec;
        float t =((point - origin) / direction).x;
        //ray->tfar = ((ray->tfar > t) & (ray->tnear < t)) ? -INFINITY : ray->tfar;

    }

}
