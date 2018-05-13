enum MatType { DIFFUSE_GLOSS = 0, REFLECT_REFRACT = 1, REFLECT = 2 };

enum AreaLightType { TRIANGLELIST = 0, SPHERELIST = 1 };

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
  __constant float *halton;
  int sphereCount;
  int triCount;
  int lightCount;
  int alightCount;
  ulong sr;
};

__constant const uint depth = 2;
#define prop_constant (2 * 2)
struct CastResult {
  float3 normal;
  float tval;
  float attent;
  struct Material mat;
  int tid;
  int sid;
};

struct RayTree {
  struct Ray r[prop_constant];
  struct CastResult res[prop_constant];
  float3 colors[prop_constant];
  int exists[prop_constant];
};

int isphere(struct Ray *r, float *t, __constant struct Sphere *sphere,
            float3 *normal) {
  float3 m = r->origin - sphere->origin;
  float b = dot(m, r->direction);
  float c = dot(m, m) - sphere->radius*sphere->radius;
  if(c > 0.0f && b > 0.0f) return -1;
  float discr = b*b - c;
  if(discr < 0.0f) return -1;
  float t0 = -b - sqrt(discr);
  *t = t0;
  float3 q = r->origin + t0*r->direction;
  *normal = normalize(q - sphere->origin);
  return 0;
}

int itriangle(struct Ray *r, float *t, __constant struct Triangle *tri,
              float3 *normal) {
  const float eps = 0.0000001;
  float3 edge1, edge2, h, s, q;
  float a, f, u, v;
  float3 pt0 = tri->pts[0];
  edge1 = tri->pts[1] - pt0;
  edge2 = tri->pts[2] - pt0;
  h = cross(r->direction, edge2);
  a = dot(edge1, h);

  if (a > -eps && a < eps) {
    return -1;
  }
  f = 1 / a;
  s = r->origin - pt0;
  u = f * dot(s, h);
  if (u < 0.0 || u > 1.0) {
    return -1;
  }
  q = cross(s, edge1);
  v = f * dot(r->direction, q);
  if (v < 0.0 || u + v > 1.0)
    return -1;
  *t = f * dot(edge2, q);
  if (*t < eps) {
    return -1;
  }
  *normal = normalize(cross(edge1, edge2));
  return 0;
}

float3 minb(float3 a, float3 b) {
  return (float3)(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

float3 maxb(float3 a, float3 b) {
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
  float3 minvec =
      minb(minb(triangle->pts[0], triangle->pts[1]), triangle->pts[2]);
  float3 maxvec =
      maxb(maxb(triangle->pts[0], triangle->pts[1]), triangle->pts[2]);
  sbound.minv.x = minvec.x - 0.0001f;
  sbound.minv.y = minvec.y - 0.0001f;
  sbound.minv.z = minvec.z - 0.0001f;
  sbound.maxv.x = maxvec.x + 0.0001f;
  sbound.maxv.y = maxvec.y + 0.0001f;
  sbound.maxv.z = maxvec.z + 0.0001f;
  return sbound;
}

struct CastResult cast(struct Ray *r, struct Scene *scene) {
  float t = 1024.0;
  float3 fcolor = {0, 0, 0};
  struct CastResult res;
  res.tval = 1024.0;
  float3 tnormal;
  for (int i = 0; i < scene->sphereCount; i++) {
    if (boundcheck(r, genSphereBounds(&scene->spheres[i])) &&
        (isphere(r, &t, &scene->spheres[i], &tnormal) != -1)) {
      if (t < res.tval) {
        res.tval = t;
        res.normal = tnormal;
        res.mat = scene->spheres[i].mat;
        res.sid = i;
        res.tid = -1;
      }
    }
  }
  for (int i = 0; i < scene->triCount; i++) {
    if (boundcheck(r, genTriangleBounds(&scene->triangles[i])) &&
        (itriangle(r, &t, &scene->triangles[i], &tnormal) != -1)) {
      if (t < res.tval) {
        res.tval = t;
        res.normal = tnormal;
        res.mat = scene->triangles[i].mat;
        res.sid = -1;
        res.tid = i;
      }
    }
  }
  return res;
}

float fresnel(float3 inc, float3 norm, float rdex) {
  float idotn = dot(inc, norm);
  float eta_i = 1.0;
  float eta_t = rdex;
  if (idotn > 0.0) {
    eta_i = eta_t;
    eta_t = 1.0;
  }

  float sin_t = eta_i / eta_t * half_sqrt(max(0.0, (1.0 - idotn * idotn)));
  if (sin_t > 1.0) {
    return 1.0;
  }
  float cos_t = half_sqrt(max((1.0 - sin_t * sin_t), 0.0));
  float cos_i = fabs(cos_i);
  float r_s =
      ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
  float r_p =
      ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));
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

uint reduce(uint x, uint N) { return ((ulong)x * (ulong)N) >> 32; }

struct CastResult reflecteval(struct CastResult ref, struct Ray *r,
                              struct Scene *scene) {
  float3 hp = r->origin + ref.tval*r->direction;
  struct CastResult refinal = ref;
  float afactor = 1.0;
  refinal.tval = 1024;
  if (refinal.mat.type == REFLECT || refinal.mat.type == REFLECT_REFRACT) {
    refinal.tid = -1;
    refinal.sid = -1;
    refinal.tval = ref.tval;
    int depth = 0;
    while (((refinal.mat.type == REFLECT) || (refinal.mat.type == REFLECT_REFRACT)) && depth < 5 &&
           (refinal.tval < 1023)) { // set depth to 0 at the first ray
      if (dot(refinal.normal, -r->direction) < 0) {
        refinal.normal = -refinal.normal;
      }
      float kr = 1.0; // fresnel(r->direction, lnormal, mat.rindex);
      afactor *= kr * 0.9;
      depth++;
      float3 refd =
          normalize(r->direction - 2.0f * dot(r->direction, refinal.normal) * refinal.normal);
      r->origin = hp + refd * 0.001f;
      r->direction = refd;
      r->inv_dir = 1.0f / refd;
      refinal = cast(r, scene);
      hp = r->origin + refinal.tval*r->direction;
    }
    if ((refinal.tval > 1023) || refinal.mat.type == REFLECT || refinal.mat.type == REFLECT_REFRACT) {
      refinal.tid = -1;
      refinal.sid = -1;
    }
  }
  refinal.attent = afactor;
  return refinal;
}
float3 refract(float3 in, float3 n, float ior) {
  float cosi = clamp(-1.0f, 1.0f, dot(in, n));
  float etai = 1, etat = ior;
  float3 ncopy = n;
  if(cosi < 0) { cosi = -cosi; } 
  else {
    float etaicopy = etai;
    etai = etat;
    etat = etai;
    ncopy = -ncopy;
  }
  float eta = etai / etat;
  float k = 1 - eta * eta * (1 - cosi * cosi);
  if(k < 0) {
    return (0, 0, 0);
  } else {
    return eta * in + (eta * cosi - sqrt(k)) * ncopy;
  }
}
struct CastResult refracteval(struct CastResult ref, struct Ray *r,
                              struct Scene *scene) {
  float3 hp = r->origin + ref.tval*r->direction;
  struct CastResult refinal = ref;
  float afactor = 1.0;
  refinal.tval = 1024;
  if (refinal.mat.type == REFLECT_REFRACT) {
    refinal.tid = -1;
    refinal.sid = -1;
    refinal.tval = ref.tval;
    int depth = 0;
    while ((refinal.mat.type == REFLECT_REFRACT) && depth < 5 &&
           (refinal.tval < 1023)) { // set depth to 0 at the first ray
      if (dot(refinal.normal, -r->direction) < 0) {
        refinal.normal = -refinal.normal;
      }
      float kr = 1.0; // fresnel(r->direction, lnormal, mat.rindex);
      afactor *= kr * 0.8;
      depth++;
      float3 refd =
          refract(r->direction, refinal.normal, refinal.mat.rindex);
      r->origin = hp + refd * 0.001f;
      r->direction = refd;
      r->inv_dir = 1.0f / refd;
      refinal = cast(r, scene);
      hp = r->origin + refinal.tval*r->direction;
    }
    if ((refinal.tval > 1023) || refinal.mat.type == REFLECT_REFRACT) {
      refinal.tid = -1;
      refinal.sid = -1;
    }
  }
  refinal.attent = afactor;
  return refinal;
}

float3 shade(struct Ray *r, struct Scene *scene, struct CastResult res) {
  float3 sky = {0.12752977781, 0.3066347662, 0.845164518};
  if((res.tid == -1) && (res.sid == -1)) {
    return sky;
  }
  if((res.mat.type == REFLECT) || (res.mat.type == REFLECT_REFRACT)) {
    return (0, 0, 0, 0);
  }
  if(res.tval > 1023) {
    return (0, 0, 0, 0);
  }
  float3 hp = r->origin + res.tval*r->direction;
  float3 lightamt = {0, 0, 0};
  float3 speccolor = {0, 0, 0};
  lightamt += res.mat.emits * res.mat.color;
  ulong a = scene->sr;
  for (int i = 0; i < scene->alightCount; i++) {
    if (dot(res.normal, -r->direction) < 0) {
      res.normal = -res.normal;
    }
    float3 specsample = {0, 0, 0};
    float3 diffsample = {0, 0, 0};
    const int samplec = 1;
    const float invs = 1.0f / samplec;
    for (uint sample = 0; sample < samplec; sample++) {
      uint red = reduce(fast_rand(&a), scene->alights[i].emitters);
      if (scene->alights[i].type == TRIANGLELIST) {
        uint chooser = scene->alights[i].emitliststart + red;
        uint id = scene->emittersets[chooser];
        __constant struct Triangle *tri = &scene->triangles[id];
        float3 e0 = tri->pts[1] - tri->pts[0];
        float3 e1 = tri->pts[2] - tri->pts[0];
        float3 n = normalize(cross(e0, e1));
        ulong selection = (fast_rand(&a) & 0x01FF);
        float rb = scene->halton[selection * 2];
        float sb = scene->halton[selection * 2 + 1];
        if ((rb + sb) >= 1.0) {
          rb = 1 - rb;
          sb = 1 - sb;
        }
        float3 pt = tri->pts[0] + rb * e0 + sb * e1;
        struct Ray s;
        s.origin = pt + n * 0.001f;
        s.direction = normalize(hp - pt);
        s.inv_dir = 1.0f / s.direction;
        // return distance(pt, hp) < 0.01f;
        if ((dot(res.normal, s.direction) > 0) || (dot(n, s.direction) < 0.001f)) {
          continue;
        }
        struct CastResult cr = cast(&s, scene);
        float3 shp = s.origin + s.direction * cr.tval;
        int sameobj = (((res.tid == cr.tid) && cr.tid != -1) ||
                       ((res.sid == cr.sid) && cr.sid != -1) );
        if (distance(shp, hp) < 0.01f && sameobj &&
            (dot(res.normal, s.direction) < 0)) {
          float3 ll = normalize(pt - hp);
          float dt = dot(res.normal, ll);
          dt = clamp(dt, 0.0f, 1.0f);
          float att = distance(hp, pt);
          att *= att;
          att = 1.0 / (1.0 + (0.0162 * att));
          // att = 1.0;
          float3 halfAngle = normalize(ll - r->direction);
          diffsample +=
              (((tri->mat.color * dt)) * att * dot(n, s.direction)) * invs;
          float ahalf = acos(dot(halfAngle, res.normal));
          float expn = ahalf / res.mat.specexp;
          expn = -(expn * expn);
          float blinn = exp(expn);
          blinn = clamp(blinn, 0.0f, 1.0f);
          if (dt == 0.0) {
            blinn = 0.0;
          }
          specsample += ((blinn * tri->mat.color) * att) * invs;
        }
      }
    }
    speccolor += specsample;
    lightamt += diffsample;
  }
  for (int i = 0; i < scene->lightCount; i++) {
    if (dot(res.normal, -r->direction) < 0) {
      res.normal = -res.normal;
    }
    float3 lightpos = scene->lights[i].pos;
    float3 l = normalize(lightpos - hp);
    float dt = dot(res.normal, l);
    dt = clamp(dt, 0.0f, 1.0f);
    float accum = 0.0;
    struct Ray s;
    s.origin = hp + res.normal * 0.001f;
    s.direction = normalize(lightpos - hp);
    s.inv_dir = 1.0f / s.direction;
    struct CastResult cr = cast(&s, scene);
    float3 shp = s.origin + cr.tval * s.direction;
    if ((cr.tval > 1023) || distance(shp, s.origin) > distance(s.origin, lightpos))
      accum += (1 / 1.0);
    if (accum > 0.0) {
      float att = distance(s.origin, lightpos);
      att *= att;
      att = 1.0 / (1.0 + (0.032 * att));
      lightamt += ((scene->lights[i].color * dt) * accum) * att;
      float3 halfAngle = normalize(l - r->direction);
      float ahalf = acos(dot(halfAngle, res.normal));
      float expn = ahalf / res.mat.specexp;
      expn = -(expn * expn);
      float blinn = exp(expn);
      blinn = clamp(blinn, 0.0f, 1.0f);
      if (dt == 0.0) {
        blinn = 0.0;
      }
      speccolor += (blinn * scene->lights[i].color * accum) * att;
    }
  }
  float3 fc = lightamt * res.mat.color * res.mat.diffc + res.mat.specc * speccolor;
  return fc;
}

void populateraytree(struct RayTree *tree, struct Scene *scene) {
  for(uint i = 1; i < depth; i++) {
    uint lfact = pown(2.0f, (float)i);
    for(uint j = 0; j < lfact; j++) {
      uint nodeid = lfact + j;
      uint upnode = (nodeid/2) - 1;
      struct CastResult refinal = tree->res[upnode];
      struct Ray r = tree->r[upnode];
      float3 hp = r.origin + refinal.tval*r.direction;
      tree->colors[nodeid - 1] = (0, 0, 0);
      if(!tree->exists[upnode])
        continue;
      if(j & 0x01) {
        if(refinal.mat.type != REFLECT_REFRACT)
          goto noeval;
        if (dot(refinal.normal, -r.direction) < 0) {
          refinal.normal = -refinal.normal;
        }
        float3 refd =
            refract(r.direction, refinal.normal, refinal.mat.rindex);
        r.origin = hp + refd * 0.001f;
        r.direction = refd;
        r.inv_dir = 1.0f / refd;
        struct CastResult ref2 = cast(&r, scene);
        tree->r[nodeid - 1] = r;
        tree->res[nodeid - 1] = ref2;
        tree->exists[nodeid - 1] = 1;
        tree->colors[nodeid - 1] = (1, 0, 1);//shade(&r, scene, ref2);
      } else {
        if((refinal.mat.type != REFLECT_REFRACT) && (refinal.mat.type != REFLECT))
          goto noeval;
        if (dot(refinal.normal, -r.direction) < 0) {
          refinal.normal = -refinal.normal;
        }
        float3 refd =
            normalize(r.direction - 2.0f * dot(r.direction, refinal.normal) * refinal.normal);
        r.origin = hp + refd * 0.001f;
        r.direction = refd;
        r.inv_dir = 1.0f / refd;
        struct CastResult ref2 = cast(&r, scene);
        tree->r[nodeid - 1] = r;
        tree->res[nodeid - 1] = ref2;
        tree->exists[nodeid - 1] = 1;
        tree->colors[nodeid - 1] = (1, 0, 1);//shade(&r, scene, ref2);
      }
      noeval:
      continue;
    }
  }
}

void solveraytree(struct RayTree *tree, struct Scene *scene) {
}

float4 trace(struct Ray *r, struct Scene *scene) {
  float4 color = {(0.12752977781), (0.3066347662), 0.845164518, 1.0};
  float3 fcolor = {0, 0, 0};
  float afactor = 1.0;
  struct CastResult primres = cast(r, scene);
  struct RayTree tree;
  tree.r[0] = *r;
  tree.res[0] = primres;
  tree.exists[0] = 1;
  tree.colors[0] = shade(r, scene, primres);
  if (primres.tval > 1023)
    return color * afactor;
  populateraytree(&tree, scene);
  float3 fc = tree.colors[2];
  float4 fcolor4 = {fc.x, fc.y, fc.z, 1.0};
  return fcolor4 * afactor;
}

__kernel void _main(__write_only image2d_t img, uint width, uint height,
                    uint tricount, uint spherecount, uint lightcount,
                    __constant struct Triangle *tris,
                    __constant struct Sphere *spheres,
                    __constant struct Light *lights, struct CameraConfig camera,
                    ulong sr, __constant struct AreaLight *arealights,
                    __constant uint *emittersets, uint alightcount,
                    __constant float *halton) {
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
  scene.halton = halton;
  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;
  // float y = (float)(get_global_id(0)) / (float)(height);
  float widthhalves = width / 160;
  for (uint i = widthhalves * (get_global_id(1));
       i < widthhalves * (get_global_id(1) + 1); i++) {
    scene.sr = sr + i * height + get_global_id(0);
    float y = get_global_id(0) / (float)height;
    float x = (float)(i) / (float)(width);
    float3 camright = cross(camera.up, camera.lookat) * ((float)width / height);
    x = x - 0.5f;
    y = y - 0.5f;
    struct Ray r;
    r.origin = camera.center;
    r.direction =
        normalize(camright * x + (camera.up * y) + camera.lookat /* + noise*/);
    r.inv_dir = 1.0f / r.direction;
    float4 color = pow(trace(&r, &scene), 1 / 2.2);
    int2 xy = {/*(int)(*/ i /* + (nvx/8192.0)) % width*/,
               /*(int)(*/ get_global_id(0) /* + (nvz/8192.0)) % height*/};
    write_imagef(img, xy, color);
  }
}
