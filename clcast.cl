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

__kernel void main(__global uchar3 *fb, uint width, uint height, uint tricount, uint spherecount, uint lightcount,
                     __constant struct Triangle *tris, __constant struct Sphere *spheres, __constant struct Light *lights, struct CameraConfig camera) {
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
	float y = (float)(get_global_id(0) / width) / (float)(height);
	float3 camright = cross(camera.up, camera.lookat) * ((float)width/height);
	x = x -0.5f;
	y = y -0.5f;
				
	struct Ray r;
	r.origin = camera.center;
	r.direction    = normalize(camright*x + (-camera.up * y) + camera.lookat);
	uchar3 color = /*raytrace(&r, &scene, 0)*/ { 255, 255, 255};
	fb[get_global_id(0)] += color;
}                                 
