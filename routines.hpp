#ifndef ROUTINES_HPP
#define	ROUTINES_HPP

#include "Mesh.hpp"
#include "Coords.hpp"

void f_poisson_jac_1d(double dt, Coords* coords, Mesh ** level);
void f_poisson_opp_1d(double dt, Coords* coords, Mesh ** level);
void f_poisson_indep_resid_1d(double dt, Coords* coords, Mesh ** level);

void f_poisson_jac_3d(double dt, Coords* coords, Mesh ** level);
void f_poisson_opp_3d(double dt, Coords* coords, Mesh ** level);
void f_poisson_indep_resid_3d(double dt, Coords* coords, Mesh ** level);

void f_schrodinger_jac_1d(double dt, Coords* coords, Mesh ** level);
void f_schrodinger_opp_1d(double dt, Coords* coords, Mesh ** level);
void f_schrodinger_indep_resid_1d(double dt, Coords* coords, Mesh ** level);

void f_schrodinger_newton_jac_noV_1d(double dt, Coords* coords, Mesh ** level);
void f_schrodinger_newton_opp_noV_1d(double dt, Coords* coords, Mesh ** level);
void f_schrodinger_newton_indep_resid_1d(double dt, Coords* coords, Mesh ** level);

void f_schrodinger_newton_jac_noPsi_1d(double dt, Coords* coords, Mesh ** level);
void f_schrodinger_newton_opp_noPsi_1d(double dt, Coords* coords, Mesh ** level);

#endif

