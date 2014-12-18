#ifndef HIERARCHY_HPP
#define	HIERARCHY_HPP
#include "Mesh.hpp"
#include "Coords.hpp"

class Hierarchy {
public:
    Coords **coords;
    Mesh ***fields;
    
    int num_l;
    int num_f;
    int num_w;
    int dim;
    int size_base[3];
    
    const static int u_i   = 0;
    const static int res_i = 1;
    const static int jac_i = 2;
    const static int s_i   = 3;
    const static int old_i = 4;
    const static double factor = 0.8;
    
    double bbox[6];
    double* change;
    double* resid;
    
    Hierarchy(int num_l_, int num_f_, int dim_, int *shape_base_, double * bbox_);
    virtual ~Hierarchy();
    
    void restrictCycle(int itt, double dt,
        void (*opp)(double dt, Coords* coords, Mesh ** level), 
        void (*jac)(double dt, Coords* coords, Mesh ** level));

    void interpolateCycle(int itt, double dt,
        void (*opp)(double dt, Coords* coords, Mesh ** level),  
        void (*jac)(double dt, Coords* coords, Mesh ** level));
    
    void relaxLevel(int itt, double dt, int lev,
        void (*opp)(double dt, Coords* coords, Mesh ** level), 
        void (*jac)(double dt, Coords* coords, Mesh ** level));
    
    void vCycle(int itt, double dt,
        void (*opp)(double dt, Coords* coords, Mesh ** level),
        void (*jac)(double dt, Coords* coords, Mesh ** level));
    
    void calcSource(int lev);
    
    void calcU(int lev);
    
    void calcResidual();
    
    void zeroSource(int lev);
    
    void print(int lev);
    
    void switchFields(int field1, int field2);
    
    void s_resid(int lev);
};

#endif

