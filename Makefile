all: poisson_1d poisson_3d schrodinger schrodinger_newton
	

main_1d_schrodinger_newton.o: main_1d_schrodinger_newton.cpp mesh.hpp
	g++ -c -Ofast main_1d_schrodinger_newton.cpp
	
main_1d_schrodinger_x.o: main_1d_schrodinger_x.cpp mesh.hpp
	g++ -c -Ofast main_1d_schrodinger_x.cpp
	
main_1d_poisson.o: main_1d_poisson.cpp mesh.hpp
	g++ -c -Ofast main_1d_poisson.cpp
	
main_3d_poisson.o: main_3d_poisson.cpp mesh.hpp
	g++ -c -Ofast main_3d_poisson.cpp	
	

Mesh.o: Mesh.cpp Mesh.hpp
	g++ -c -Ofast Mesh.cpp

Coords.o: Coords.cpp Coords.hpp
	g++ -c -Ofast Coords.cpp

Hierarchy.o: Hierarchy.cpp Hierarchy.hpp
	g++ -c -Ofast Hierarchy.cpp

routines.o: routines.cpp routines.hpp
	g++ -c -Ofast routines.cpp

schrodinger_newton: main_1d_schrodinger_newton.o Mesh.o Coords.o Hierarchy.o routines.o
	g++ main_1d_schrodinger_newton.o Mesh.o Coords.o Hierarchy.o routines.o -Ofast -o schrodinger_newton

schrodinger: main_1d_schrodinger_x.o Mesh.o Coords.o Hierarchy.o routines.o
	g++ main_1d_schrodinger_x.o Mesh.o Coords.o Hierarchy.o routines.o -Ofast -o schrodinger

poisson_1d: main_1d_poisson.o Mesh.o Coords.o Hierarchy.o routines.o
	g++ main_1d_poisson.o Mesh.o Coords.o Hierarchy.o routines.o -Ofast -o poisson_1d

poisson_3d: main_3d_poisson.o Mesh.o Coords.o Hierarchy.o routines.o
	g++ main_3d_poisson.o Mesh.o Coords.o Hierarchy.o routines.o -Ofast -o poisson_3d

clean:
	rm schrodinger_newton schrodinger poisson_1d poisson_3d *.o
