
CXXFLAGS_MAIN  = -Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0
CXXFLAGS_IPNSW = -std=c++11 -mtune=native -march=native -Wall -Wunused-result -O3 -DNDEBUG -g -fopenmp -ffast-math

all: main ipnsw brute_force

main: main.cpp *.h
	g++ $(CXXFLAGS) $(CXXFLAGS_MAIN) -o main main.cpp

ipnsw: ipnsw.cpp
	g++ $(CXXFLAGS) $(CXXFLAGS_IPNSW) -o ipnsw ipnsw.cpp

brute_force: brute_force.cpp
	g++ $(CXXFLAGS) $(CXXFLAGS_IPNSW) -o brute_force brute_force.cpp

clean:
	rm -f main ipnsw