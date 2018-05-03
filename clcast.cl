enum MatType {
    DIFFUSE_GLOSS = 0,
    REFLECT_REFRACT = 1,
    REFLECT = 2
};

enum AreaLightType {
    TRIANGLELIST = 0,
    SPHERELIST = 1
};

struct Material {
    float3 color;
    enum MatType type;
    float rindex;
    float diffc;
    float specc;
    float specexp;
    float emits;
};

struct AreaLight {
    uint emitliststart;
    uint emitters;
    enum AreaLightType type;
};

struct CameraConfig {
    float3 center;
    float3 lookat;
    float3 up;
};

struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_dir;
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

struct AABB {
    float3 minv;
    float3 maxv;
};
/*
struct AABB[2] splitX(struct AABB bounds) {
    struct AABB splits[2];
    splits[0].minv = bounds.minv;
    float3 nmaxv0 = bounds.maxv;
    nmaxv0.x /= 2;
    splits[0].maxv = nmaxv0;
    splits[1].minv = nmaxv0;
    splits[1].maxv = bounds.maxv;
    return splits;
}

struct AABB[2] splitY(struct AABB bounds) {
    struct AABB splits[2];
    splits[0].minv = bounds.minv;
    float3 nmaxv0 = bounds.maxv;
    nmaxv0.y /= 2;
    splits[0].maxv = nmaxv0;
    splits[1].minv = nmaxv0;
    splits[1].maxv = bounds.maxv;
    return splits;
}

struct AABB[2] splitZ(struct AABB bounds) {
    struct AABB splits[2];
    splits[0].minv = bounds.minv;
    float3 nmaxv0 = bounds.maxv;
    nmaxv0.z /= 2;
    splits[0].maxv = nmaxv0;
    splits[1].minv = nmaxv0;
    splits[1].maxv = bounds.maxv;
    return splits;
}*/

struct Scene {
    __constant struct Sphere *spheres;
    __constant struct Triangle *triangles;
    __constant struct Light *lights;
    __constant struct AreaLight *alights;
    __constant uint *emittersets;
    int sphereCount;
    int triCount;
    int lightCount;
    int alightCount;
    ulong sr;
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
    float h =  half_sqrt(rpow2 - apow2);
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

float3 minb(float3 a, float3 b)
{
    return (float3)(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

float3 maxb(float3 a, float3 b)
{
    return (float3)(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

int boundcheck(struct Ray *r, struct AABB bound) {
    float3 v1 = (bound.minv - r->origin) * r->inv_dir;
    float3 v2 = (bound.maxv - r->origin) * r->inv_dir;
    float3 n = minb(v1, v2);
    float3 f = maxb(v1, v2);
    float enter = max(n.x, max(n.y, n.z));
    float exit = min(f.x, min(f.y, f.z));
    return (exit > 0.0f && enter < exit);
}

struct AABB genSphereBounds(__constant struct Sphere *sphere) {
    struct AABB sbound;
    sbound.minv.x = sphere->origin.x - sphere->radius; 
    sbound.minv.y = sphere->origin.y - sphere->radius; 
    sbound.minv.z = sphere->origin.z - sphere->radius;
    sbound.maxv.x = sphere->origin.x + sphere->radius;
    sbound.maxv.y = sphere->origin.y + sphere->radius;
    sbound.maxv.z = sphere->origin.z + sphere->radius;
    return sbound;
}

struct AABB genTriangleBounds(__constant struct Triangle *triangle) {
    struct AABB sbound;
    float3 minvec = minb(minb(triangle->pts[0], triangle->pts[1]), triangle->pts[2]);
    float3 maxvec = maxb(maxb(triangle->pts[0], triangle->pts[1]), triangle->pts[2]);
    sbound.minv.x = minvec.x - 0.0001f; 
    sbound.minv.y = minvec.y - 0.0001f; 
    sbound.minv.z = minvec.z - 0.0001f;
    sbound.maxv.x = maxvec.x + 0.0001f; 
    sbound.maxv.y = maxvec.y + 0.0001f; 
    sbound.maxv.z = maxvec.z + 0.0001f;
    return sbound;
}


void cast(struct Ray* r, struct Scene* scene, float3 *normal, float *spherelowt, struct Material *mat, int *tp, int *sp) {
    float t = 1024.0;
    float3 fcolor= { 0,0,0};
    float3 tnormal;
    for(int i = 0; i < scene->sphereCount; i++) {
        if(boundcheck(r, genSphereBounds(&scene->spheres[i])) && (isphere(r, &t, &scene->spheres[i], &tnormal)  != -1)) {
            if(t < *spherelowt) {
                *spherelowt = t;
                *normal = tnormal;
                *mat = scene->spheres[i].mat;
                *sp = i;
                *tp = -1;
            }
        }
    }
    for(int i = 0; i < scene->triCount; i++) {
        if(boundcheck(r, genTriangleBounds(&scene->triangles[i])) && (itriangle(r, &t, &scene->triangles[i], &tnormal)  != -1)) {
            if(t < *spherelowt) {
                *spherelowt = t;
                *normal = tnormal;
                *mat = scene->triangles[i].mat;
                *sp = -1;
                *tp = i;
            }
        }
    }
}

float fresnel(float3 inc, float3 norm, float rdex) {
    float idotn = dot(inc, norm);
    float eta_i = 1.0;
    float eta_t = rdex;
    if(idotn > 0.0) {
        eta_i = eta_t;
        eta_t = 1.0;
    }
    
    float sin_t = eta_i / eta_t * half_sqrt(max(0.0, (1.0 - idotn*idotn)));
    if(sin_t > 1.0) {
        return 1.0;
    }
    float cos_t = half_sqrt(max((1.0 - sin_t * sin_t), 0.0));
    float cos_i = fabs(cos_i);
    float r_s = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
    float r_p = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));
    return (r_s * r_s + r_p * r_p) / 2.0;
}

inline long fast_rand(ulong *x) {
    ulong z = (*x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

inline float3 randUnitVec(ulong *r) {
    float3 vec;
    vec.x = fast_rand(r);
    vec.y = fast_rand(r);
    vec.z = fast_rand(r);
    return normalize(vec);
}

uint reduce(uint x, uint N) {
  return ((ulong) x * (ulong) N) >> 32 ;
}

float4 trace(struct Ray *r, struct Scene* scene) {
    float4 color = {(0.12752977781), (0.3066347662), 0.845164518, 1.0};
    float spherelowt = 1024.0;
    float3 fcolor= { 0,0,0};
    float3 lnormal;
    float3 tnormal;
    float afactor = 1.0;
    struct Material mat;
    int trash;
    int trash2;
    cast(r, scene, &lnormal, &spherelowt, &mat, &trash2, &trash);
    float3 hp = r->origin + spherelowt*r->direction;
    if(spherelowt > 1023)
        return color * afactor;
    //return (scene->sphereCount, scene->sphereCount, scene->sphereCount, scene->sphereCount);
    if (mat.type == REFLECT) {
        int depth = 0;
	    while((mat.type == REFLECT) && depth < 3) { //set depth to 0 at the first ray           
            if(dot(lnormal, -r->direction) < 0) {
                lnormal = -lnormal;
            }
            float kr = 1.0; //fresnel(r->direction, lnormal, mat.rindex);
		    afactor *= kr * 0.9;  
		    depth++;
		    float3 refd = normalize(r->direction - 2.0f*dot(r->direction, lnormal)*lnormal);
		    r->origin = hp + refd*0.001f;
		    r->direction = refd;
            r->inv_dir = 1.0f/refd;
		    spherelowt = 1024;  
            ulong ptr;
		    cast(r, scene, &lnormal, &spherelowt, &mat, &trash2, &trash);
		    if(spherelowt > 1023)
		        break;
		    hp = r->origin + spherelowt*r->direction;
	    }
	    if((spherelowt > 1023) || (mat.type == REFLECT))
		    return color * afactor;
    }
    float3 lightamt = {0, 0, 0};
    float3 speccolor = {0, 0, 0};
    lightamt += mat.emits * mat.color;
    float x, y, z;
    x = r->direction.x;
    y = r->direction.y;
    z = r->direction.z;
    ulong a = scene->sr;
    for(int i = 0; i < scene->alightCount; i++) {
        int savedtrash = trash2;
        if(dot(lnormal, -r->direction) < 0) {
            lnormal = -lnormal;
        }
        float3 specsample = {0, 0, 0};
        float3 diffsample = {0, 0, 0};
        for(uint sample = 0; sample < 64; sample++) {
            uint red = reduce(fast_rand(&a), scene->alights[i].emitters);
            if(scene->alights[i].type == TRIANGLELIST) {
                uint chooser = scene->alights[i].emitliststart + red;
                uint id = scene->emittersets[chooser];
                __constant struct Triangle *tri = &scene->triangles[id];
                float3 e0 = tri->pts[1] - tri->pts[0];
                float3 e1 = tri->pts[2] - tri->pts[0];
                float3 n = normalize(cross(e0, e1));
                float rb = (fast_rand(&a) & 0xFFFF)/65535.0f;
                float sb = (fast_rand(&a) & 0xFFFF)/65535.0f;
                if((rb+sb) >= 1.0) {
                    rb = 1 - rb;
                    sb = 1 - sb;
                }
                float3 pt = tri->pts[0] + rb*e0 + sb*e1;
                struct Ray s;
                s.origin = pt + n*0.001f;
                s.direction = normalize(hp - pt);
                s.inv_dir = 1.0f/s.direction;
		if((dot(lnormal, s.direction) > 0) || (dot(n, s.direction) < 0.001f)) {
			continue;
		}
                float4 vv = {s.direction.x, s.direction.y, s.direction.z, 1.0};
                float t = 1024;
                float3 sn;
                struct Material c;
                ulong ptr;
                cast(&s, scene, &sn, &t, &c, &trash2, &trash);
                float3 shp = s.origin + s.direction*t;
                float accum = 0.0;
                if(distance(shp, hp) < 0.01f && (trash2 == savedtrash) && (dot(lnormal, s.direction) < 0)) {
                    accum += 1.0;
                }
                if(accum > 0.0) {
                    float dt = dot(lnormal, pt);
                    dt = clamp(dt, 0.0f, 1.0f);
                    float att = distance(hp, pt);
                    att *= att;
                    att = 1.0 / (1.0 + (0.0162 * att));
                    //att = 1.0;
                    float3 halfAngle = normalize(pt - r->direction);
                    diffsample += (((tri->mat.color) * accum) * att * dot(n, s.direction))/(64);
                    float ahalf = acos(dot(halfAngle, lnormal));
                    float expn = ahalf / mat.specexp;
                    expn = -(expn*expn);
                    float blinn = exp(expn);
                    blinn = clamp(blinn, 0.0f, 1.0f);
                    if(dt == 0.0) {
                         blinn = 0.0;
                    }
                    specsample += ((blinn * tri->mat.color * accum) * att)/(64);
                }
            }
        }
        speccolor += specsample;
        lightamt += diffsample;
    }
    for(int i = 0; i < scene->lightCount; i++) {
        if(dot(lnormal, -r->direction) < 0) {
            lnormal = -lnormal;
        }
        float3 lightpos = scene->lights[i].pos;
        float3 l = normalize(lightpos - hp);
        float dt = dot(lnormal, l);
        dt = clamp(dt, 0.0f, 1.0f);
        float accum = 0.0;
        struct Ray s;
        s.origin = hp + lnormal*0.001f;
        s.direction  = normalize(lightpos - hp);
        s.inv_dir = 1.0f/s.direction;
        float t = 1024;
        float3 sn;
        struct Material c;
        ulong ptr;
        cast(&s, scene, &sn, &t, &c, &trash2, &trash);
        float3 shp = s.origin + t*s.direction;
        if((t > 1023) || distance(shp, s.origin) > distance(s.origin, lightpos)) 
            accum += (1/1.0);
        if(accum > 0.0) {
            float att = distance(s.origin, lightpos);
            att *= att;
            att = 1.0 / (1.0 + (0.032 * att));
            lightamt += ((scene->lights[i].color*dt) * accum) * att;
            float3 halfAngle = normalize(l - r->direction);
            float ahalf = acos(dot(halfAngle, lnormal));
            float expn = ahalf / mat.specexp;
            expn = -(expn*expn);
            float blinn = exp(expn);
            blinn = clamp(blinn, 0.0f, 1.0f);
            if(dt == 0.0) {
                 blinn = 0.0;
            }
            speccolor += (blinn * scene->lights[i].color * accum) * att;
        }
    }
    float3 fc = lightamt * mat.color * mat.diffc + mat.specc * speccolor;
    float4 fcolor4 = {fc.x, fc.y, fc.z, 1.0};
    return fcolor4 * afactor;
}

__kernel void _main(__write_only image2d_t img, uint width, uint height, uint tricount, uint spherecount, uint lightcount, __constant struct Triangle *tris, __constant struct Sphere *spheres, __constant struct Light *lights, struct CameraConfig camera, ulong sr, __constant struct AreaLight *arealights, __constant uint *emittersets, uint alightcount) {
    struct Scene scene;
    scene.triangles = tris;
    scene.spheres = spheres;
    scene.lights = lights;
    scene.sphereCount = spherecount;
    scene.triCount = tricount;
    scene.lightCount = lightcount;
    scene.alightCount = alightcount;
    scene.alights = arealights;
    scene.emittersets = emittersets;
    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;
    //float y = (float)(get_global_id(0)) / (float)(height);
    float widthhalves = width/16;
    for(uint i = widthhalves*(get_global_id(1)); i < widthhalves*(get_global_id(1)+1); i++) {
            scene.sr = sr + i*height + get_global_id(0);
            float y = get_global_id(0)/(float)height;
            float x = (float)(i) / (float)(width);
            float3 camright = cross(camera.up, camera.lookat) * ((float)width/height);
            x = x -0.5f;
            y = y -0.5f;				
            struct Ray r;
            r.origin = camera.center;
            r.direction    = normalize(camright*x + (camera.up * y) + camera.lookat/* + noise*/);
            r.inv_dir = 1.0f/r.direction;
            float4 color = pow(trace(&r, &scene), 1 / 2.2);
            int2 xy = {/*(int)(*/i/* + (nvx/8192.0)) % width*/, /*(int)(*/get_global_id(0)/* + (nvz/8192.0)) % height*/};
            write_imagef(img, xy, color);
    }
}                                 
