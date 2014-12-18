#ifndef COORDS_HPP
#define	COORDS_HPP
#include "Mesh.hpp"

class Coords
{
public:
    double delta[3];
    double bbox[6];
    Mesh **mesh;
    int size[3];
    int dim;
    
    Coords(int dim_, int* size_, double* bbox_);
    virtual ~Coords();
};

#endif

