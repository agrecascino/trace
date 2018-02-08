struct Material {
    float3 color;
    int reflective;
};

struct CameraConfig {
    float3 center;
    float3 lookat;
    float3 up;
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct Light {
    float3 pos;
    float3 color;
};

struct Sphere {
    struct Material mat;
    float3 origin;
    float radius;
};

struct Triangle {
    struct Material mat;
    float3 pts[3];
};

struct Scene {
    __constant struct Sphere *spheres;
    __constant struct Triangle *triangles;
    __constant struct Light *lights;
    int sphereCount;
    int triCount;
    int lightCount;
};


int isphere(struct Ray *r, float *t, __constant struct Sphere *sphere, float3 *normal) {
    float3 p = r->origin - sphere->origin;
    float rpow2 = sphere->radius*sphere->radius;
    float p_d = dot(p, r->direction);

    if(p_d > 0 || dot(p, p) < rpow2)
        return -1;

    float3 a = p - p_d * r->direction;
    float apow2 = dot(a, a);
    if(apow2 > rpow2)
        return -1;
    float h = sqrt(rpow2 - apow2);
    float3 i = a - h*r->direction;
    float3 pt = sphere->origin + i;
    *t = ((pt-r->origin) / r->direction).x;
    *normal = i/sphere->radius;
    return 0;
    
}

int itriangle(struct Ray *r, float *t, __constant struct Triangle *tri, float3 *normal) {
    const float eps = 0.0000001;
    float3 edge1, edge2, h, s, q;
    float a, f, u, v;
    float3 pt0 = tri->pts[0];
    edge1 = tri->pts[1] - pt0;
    edge2 = tri->pts[2] - pt0;
    h = cross(r->direction, edge2);
    a = dot(edge1, h);

    if(a > -eps && a < eps) {
        return -1;
    }
    f = 1/a;
    s = r->origin - pt0;
    u = f * dot(s, h);
    if(u < 0.0 || u > 1.0) {
        return -1;
    }
    q = cross(s, edge1);
    v = f * dot(r->direction, q);
    if(v < 0.0 || u+v > 1.0)
        return -1;
    *t = f * dot(edge2, q);
    if(*t < eps) {
        return -1;
    }
    *normal = normalize(cross(edge1, edge2));
    return 0;
}

void cast(struct Ray* r, struct Scene* scene, float3 *normal, float *spherelowt, float3 *color2, bool *reflected) {
    float t = 1024.0;
    float3 fcolor= { 0,0,0};
    float3 tnormal;
    for(int i = 0; i < scene->sphereCount; i++) {
        if(isphere(r, &t, &scene->spheres[i], &tnormal)  != -1) {
            if(t < *spherelowt) {
                *spherelowt = t;
                color2->x = scene->spheres[i].mat.color.x;
                color2->y = scene->spheres[i].mat.color.y;
                color2->z = scene->spheres[i].mat.color.z;
                *normal = tnormal;
                *reflected = scene->triangles[i].mat.reflective;
            }
        }
    }
    for(int i = 0; i < scene->triCount; i++) {
        if(itriangle(r, &t, &scene->triangles[i], &tnormal)  != -1) {
            if(t < *spherelowt) {
                *spherelowt = t;
                color2->x = scene->triangles[i].mat.color.x;
                color2->y = scene->triangles[i].mat.color.y;
                color2->z = scene->triangles[i].mat.color.z;
                *normal = tnormal;
                *reflected = scene->triangles[i].mat.reflective;
            }
        }
    }
}

float4 trace(struct Ray *r, struct Scene* scene) {
    float4 color = { 100.0/255, 149.0/255, 247.0/255, 1.0};
    float3 color2;
    float spherelowt = 1024.0;
    float3 fcolor= { 0,0,0};
    float3 lnormal;
    float3 tnormal;
    float afactor = 1.0;
    bool reflect = false;
    cast(r, scene, &lnormal, &spherelowt, &color2, &reflect);
    float3 hp = r->origin + spherelowt*r->direction;
    if(dot(lnormal, -r->direction) < 0) {
        lnormal = -lnormal;
    }
    if(spherelowt > 1023)
        return color * afactor;
    //fcolor = color2;
    for(int i = 0; i < scene->lightCount; i++) {
        float3 lightpos = scene->lights[i].pos;
        float3 l = lightpos - hp;
        float dt = dot(normalize(l), lnormal);
        struct Ray s;
        s.origin = hp + lnormal*0.001f;
        s.direction  = normalize(lightpos - hp);
        float t = 1024;
        float3 sn;
        float3 c;
        bool r = false;
        cast(&s, scene, &sn, &t, &c, &r);
        float3 shp = s.origin + t*s.direction;
        if((t > 1023) || distance(shp, s.origin) > distance(s.origin, lightpos))
            fcolor += (scene->lights[i].color*dt)*color2;
    }
    float4 fcolor4 = {fcolor.x, fcolor.y, fcolor.z, 1.0};
    return fcolor4 * afactor;
}

__kernel void _main(__write_only image2d_t img, uint width, uint height, uint tricount, uint spherecount, uint lightcount, __constant struct Triangle *tris, __constant struct Sphere *spheres, __constant struct Light *lights, struct CameraConfig camera) {
    struct Scene scene;
    scene.triangles = tris;
    scene.spheres = spheres;
    scene.lights = lights;
    scene.sphereCount = spherecount;
    scene.triCount = tricount;
    scene.lightCount = lightcount;
    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;
    float x = (float)(get_global_id(0) % width) / (float)(width);
	
    float y = (float)(get_global_id(1)) / (float)(height);	
    float3 camright = cross(camera.up, camera.lookat) * ((float)width/height);
    x = x -0.5f;
    y = y -0.5f;				
    struct Ray r;
    r.origin = camera.center;   
    r.direction    = normalize(camright*x + (camera.up * y) + camera.lookat);
    float4 color = trace(&r, &scene);
    //float4 color = { scene.lights[0].pos.x, scene.lights[0].pos.y, scene.lights[0].pos.z, 1.0 };
    //float4 color = { 1, 1, 1, 1};
    int2 xy = {get_global_id(0), get_global_id(1)};
    write_imagef(img, xy, color);
}                                 
