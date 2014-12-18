#ifndef MESH_HPP
#define	MESH_HPP
#define MIN_GRID_SIZE 100000

class Mesh {
public:
    int dim;
    int size[3];
    int n;
    double* data;
    double* f1d;
    double** f2d;
    double*** f3d;

    Mesh(int dim_, int* size_);
    virtual ~Mesh();

    static int restrict(Mesh* c, Mesh* f);
    static int interpolate(Mesh* c, Mesh* f);
    static void print(Mesh* mesh);
    
private:
    static int interpolate1d(Mesh* c, Mesh* f);
    static int interpolate2d(Mesh* c, Mesh* f);
    static int interpolate3d(Mesh* c, Mesh* f);
    
    static int restrict1d(Mesh* c, Mesh* f);
    static int restrict2d(Mesh* c, Mesh* f);
    static int restrict3d(Mesh* c, Mesh* f);
    
    static void print1d(Mesh* mesh);
    static void print2d(Mesh* mesh);
    static void print3d(Mesh* mesh);
};

#endif
