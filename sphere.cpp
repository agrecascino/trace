#include "sphere.h"

Sphere::Sphere(glm::vec3 origin, float radius, Material mat) : origin(origin), radius(radius), mat(mat) {}
glm::vec3 Sphere::getNormal(Ray &ray, float &t) { return ((ray.origin + (t*ray.direction) - origin)) / radius; }

Material Sphere::getMaterial() {
    return mat;
}

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
    ret.mat = mat;
    return ret;
}

IntersectableInternalOptimized* Sphere::optimize() {
    SphereInternalOptimized *o = new SphereInternalOptimized;
    o->mat = mat;
    o->type = SphereType;
    o->origin = origin;
    o->radius = radius;
    return (IntersectableInternalOptimized*)o;
}

unsigned int Sphere::getGeomID() const {
    return geomID;
}

void Sphere::setGeomID(unsigned int id) {
    geomID = id;
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
    if (args->context == nullptr)
        return;
    void* ptr  = args->geometryUserPtr;
    RTCRayN* rays = (RTCRayN*)args->ray;
    unsigned int n = args->N;
    for(size_t i = 0; i < n; i++) {
        if (valid[i] != -1) continue;
        glm::vec3 origin = glm::vec3(RTCRayN_org_x(rays, n, i), RTCRayN_org_y(rays, n, i), RTCRayN_org_z(rays, n, i));
        glm::vec3 direction = glm::vec3(RTCRayN_dir_x(rays, n, i), RTCRayN_dir_y(rays, n, i), RTCRayN_dir_z(rays, n, i));
        const Sphere *sphere = (Sphere*)ptr;
        glm::vec3 p = origin - sphere->origin;
        float rpow2 = sphere->radius*sphere->radius;
        float p_d = glm::dot(p, direction);
        if(p_d > 0 || dot(p, p) < rpow2)
            continue;
        glm::vec3 a = p - p_d * direction;
        float apow2 = glm::dot(a, a);
        if(apow2 > rpow2)
            continue;
        float h = sqrtf(rpow2 - apow2);
        glm::vec3 iV = a - h*direction;
        glm::vec3 point = origin + iV;
        float t = ((point - origin) / direction).x;
        RTCRayN_tfar(rays, n, i) = ((RTCRayN_tfar(rays, n, i) > t) & (RTCRayN_tnear(rays, n, i) < t)) ? -INFINITY : t;
    }
}


void sphereIntersectFunc(const RTCIntersectFunctionNArguments* args) {
    //FIXME: fix faster sphere-ray collison.
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
        const glm::vec3 m = origin - sphere->origin;
        const float b = glm::dot(m, direction);
        const float c = glm::dot(m,m) - (sphere->radius*sphere->radius);
        if(c > 0.0f && b > 0.0f)
            continue;
        float discr = b*b - c;
        if(discr < 0.0f)
            continue;
        float t = -b - sqrtf(discr);
//        glm::vec3 p = origin - sphere->origin;
//        float rpow2 = sphere->radius*sphere->radius;
//        float p_d = glm::dot(p, direction);
//        if(p_d > 0 || dot(p, p) < rpow2)
//            continue;
//        glm::vec3 a = p - p_d * direction;
//        float apow2 = glm::dot(a, a);
//        if(apow2 > rpow2)
//            continue;
//        float h = sqrtf(rpow2 - apow2);
//        glm::vec3 i_vec = a - h*direction;
//        glm::vec3 point = origin + i_vec;
//        float t = ((point - origin) / direction).x;
        RTCHitN_primID(hits, n, i) = args->primID;
        RTCHitN_u(hits, n, i) = 0.0;
        RTCHitN_v(hits, n, i) = 0.0;
        RTCRayN_tfar(rays, n, i) =  t;
        RTCHitN_geomID(hits, n, i) = sphere->getGeomID();
        //ray->tfar = ((ray->tfar > t) & (ray->tnear < t)) ? -INFINITY : ray->tfar;

    }

}
