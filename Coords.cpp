#include "Coords.hpp"
#include <stdio.h>

/*
 * Represents a set of coordinates in a Hiearchy instance.
 * dim_:  dimension of the Hiearchy (1,2,3)
 * size_: array of length dim_ containing the number of grid points in each 
 *        dimension
 * bbox_: array of length 2*dim_ containing the ordered boundaries of the 
 *        region. For example, a 2D grid defined on 0<x<1, 1<y<10 
 *        bbox_={0,1,1,10}
 */
Coords::Coords(int dim_, int* size_, double* bbox_){
   int i, j;
   dim = dim_;
   mesh = new Mesh*[dim];
   int coord_size[1];

   for (i = 0; i < dim; i++)
   {
      size[i] = size_[i];
      bbox[2*i] = bbox_[2*i];
      bbox[2*i+1] = bbox_[2*i+1];
      delta[i] = (bbox[2*i+1] - bbox[2*i]) / ((double) (size[i] - 1));
      
      coord_size[0] = size[i];
      mesh[i] = new Mesh(1, coord_size);
      for (j = 0; j < size[i]; j++)
      {
         mesh[i]->f1d[j] = bbox[2*i] + j*delta[i];
      }
   }
}

Coords::~Coords() {
   int i;
   for (i = 0; i < dim; i++)
   {
      delete mesh[i];
   }
   delete mesh;
}