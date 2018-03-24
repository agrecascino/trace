enum MatType {
    DIFFUSE_GLOSS,
    REFLECT_REFRACT,
    REFLECT
};

struct Material {
    float3 color;
    enum MatType type;
    float rindex;
    float diffc;
    float specc;
    float specexp;
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

void cast(struct Ray* r, struct Scene* scene, float3 *normal, float *spherelowt, struct Material *mat) {
    float t = 1024.0;
    float3 fcolor= { 0,0,0};
    float3 tnormal;
    for(int i = 0; i < scene->sphereCount; i++) {
        if(isphere(r, &t, &scene->spheres[i], &tnormal)  != -1) {
            if(t < *spherelowt) {
                *spherelowt = t;
                *normal = tnormal;
                *mat = scene->spheres[i].mat;
            }
        }
    }
    for(int i = 0; i < scene->triCount; i++) {
        if(itriangle(r, &t, &scene->triangles[i], &tnormal)  != -1) {
            if(t < *spherelowt) {
                *spherelowt = t;
                *normal = tnormal;
                *mat = scene->triangles[i].mat;
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
    
    float sin_t = sqrt(max(0.0, (eta_i / eta_t * (1.0 - idotn*idotn))));
    if(sin_t > 1.0) {
        return 1.0;
    }
    float cos_t = sqrt(max((1.0 - sin_t * sin_t), 0.0));
    float cos_i = fabs(cos_t);
    float r_s = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
    float r_p = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));
    return (r_s * r_s + r_p * r_p) / 2.0;
}

float4 trace(struct Ray *r, struct Scene* scene) {
    float4 color = {(100/255.0), (149/255.0), (237/255.0), 1.0};
    float spherelowt = 1024.0;
    float3 fcolor= { 0,0,0};
    float3 lnormal;
    float3 tnormal;
    float afactor = 1.0;
    struct Material mat;
    cast(r, scene, &lnormal, &spherelowt, &mat);
    float3 hp = r->origin + spherelowt*r->direction;
    if(spherelowt > 1023)
        return color * afactor;
    if (mat.type == REFLECT) {
        int depth = 0;
	    while((mat.type == REFLECT) && depth < 3) { //set depth to 0 at the first ray           
            if(dot(lnormal, -r->direction) < 0) {
                lnormal = -lnormal;
            }
            float kr = fresnel(r->direction, lnormal, mat.rindex);
		    afactor *= kr * 0.9;  
		    depth++;
		    float3 refd = normalize(r->direction - 2.0f*dot(r->direction, lnormal)*lnormal);
		    r->origin = hp + refd*0.001f;
		    r->direction = refd;
		    spherelowt = 1024;  
		    cast(r, scene, &lnormal, &spherelowt, &mat);
		    if(spherelowt > 1023)
		        break;
		    hp = r->origin + spherelowt*r->direction;
	    }
	    if((spherelowt > 1023) || (mat.type == REFLECT))
		    return color * afactor;
    }
    float3 lightamt = {0, 0, 0};
    float3 speccolor = {0, 0, 0};
    for(int i = 0; i < scene->lightCount; i++) {
        if(dot(lnormal, -r->direction) < 0) {
            lnormal = -lnormal;
        }
        float3 lightpos = scene->lights[i].pos;
        float3 l = normalize(lightpos - hp);
        float dt = dot(l, lnormal);
        struct Ray s;
        s.origin = hp + lnormal*0.001f;
        s.direction  = normalize(lightpos - hp);
        float t = 1024;
        float3 sn;
        struct Material c;
        cast(&s, scene, &sn, &t, &c);
        float3 shp = s.origin + t*s.direction;
        if((t > 1023) || distance(shp, s.origin) > distance(s.origin, lightpos)) {
            float3 refd = normalize(-l - 2.0f*dot(-l, lnormal)*lnormal);
            lightamt += (scene->lights[i].color*dt); 
            speccolor += pow(max(0.0f, -dot(refd, r->direction)), mat.specexp) * scene->lights[i].color;
        }
    }
    float3 fc = lightamt * mat.color * mat.diffc + mat.specc * speccolor;
    float4 fcolor4 = {fc.x, fc.y, fc.z, 1.0};
    return fcolor4 * afactor;
}

inline int fast_rand(uint g_seed) {
    g_seed = (214013*g_seed+2531011);
    return (g_seed>>16)&0x7FFF;
}

__kernel void _main(__write_only image2d_t img, uint width, uint height, uint tricount, uint spherecount, uint lightcount, __constant struct Triangle *tris, __constant struct Sphere *spheres, __constant struct Light *lights, struct CameraConfig camera, uint sr) {
    struct Scene scene;
    scene.triangles = tris;
    scene.spheres = spheres;
    scene.lights = lights;
    scene.sphereCount = spherecount;
    scene.triCount = tricount;
    scene.lightCount = lightcount;
    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;
    float y = (float)(get_global_id(0)) / (float)(height);	
    for(uint i = 0; i < width; i++) {
            float x = (float)(i % width) / (float)(width);
            float3 camright = cross(camera.up, camera.lookat) * ((float)width/height);
            x = x -0.5f;
            y = y -0.5f;				
            struct Ray r;
            r.origin = camera.center;   
            //uint nvx  =  fast_rand(sr+get_global_id(0)*width + get_global_id(1));
            //uint nvy  =  fast_rand(nvx);
            //uint nvz  =  fast_rand(nvy);
            //float3 noise = { ((int)nvx-16384.0)/655360.0f, ((int)nvy-16384.0)/655360.0f, ((int)nvz-16384.0)/655360.0f};
            r.direction    = normalize(camright*x + (camera.up * y) + camera.lookat/* + noise*/);
            float4 color = trace(&r, &scene);
            //float4 color = { scene.lights[0].pos.x, scene.lights[0].pos.y, scene.lights[0].pos.z, 1.0 };
            //float4 color = { 1, 1, 1, 1};
            int2 xy = {/*(int)(*/i/* + (nvx/8192.0)) % width*/, /*(int)(*/get_global_id(0)/* + (nvz/8192.0)) % height*/};
            write_imagef(img, xy, color);
    }
}                                 
