#pragma GCC diagnostic ignored "-Wunused-result"

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string.h>
#include <omp.h>

#include <cmath>
#include <queue>
#include <vector>
#include <set>

using namespace std;
using namespace std::chrono;
// #define VERBOSE

// --
// Global defs

typedef int Int;
typedef float Real;

// params

unsigned int n_runs = 1;
unsigned int ef     = 128;

const Int v_entry   = 82026;
const Int n_results = 10;

// vectors
Int data_row, data_dim;
Real* data;

Int queries_row, queries_dim;
Real* queries;

// graphs
Int G0_row, G0_col, G0_nnz;
Int* G0_rowptr;
Int* G0_columns;
Real* G0_csr_data;

Int G1_row, G1_col, G1_nnz;
Int* G1_rowptr;
Int* G1_columns;
Real* G1_csr_data;

Int G2_row, G2_col, G2_nnz;
Int* G2_rowptr;
Int* G2_columns;
Real* G2_csr_data;

Int G3_row, G3_col, G3_nnz;
Int* G3_rowptr;
Int* G3_columns;
Real* G3_csr_data;

// output
Int* scores;

// --
// IO

void load_data(std::string inpath) {
    FILE* ptr = fopen(inpath.c_str(), "rb");

    // Read `data`
    fread(&data_row, sizeof(Int), 1, ptr);
    fread(&data_dim, sizeof(Int), 1, ptr);

    data = (Real*)malloc(data_row * data_dim * sizeof(Real));
    fread(data, sizeof(Real), data_row * data_dim, ptr);

    // Read `queries`
    fread(&queries_row, sizeof(Int), 1, ptr);
    fread(&queries_dim, sizeof(Int), 1, ptr);

    queries = (Real*)malloc(queries_row * queries_dim * sizeof(Real));
    fread(queries, sizeof(Real), queries_row * queries_dim, ptr);

    // Read adjacency matrices
    fread(&G0_row, sizeof(Int), 1, ptr);
    fread(&G0_col, sizeof(Int), 1, ptr);
    fread(&G0_nnz, sizeof(Int), 1, ptr);
    G0_rowptr   = (Int*)malloc((G0_row + 1) * sizeof(Int));
    G0_columns  = (Int*)malloc(G0_nnz * sizeof(Int));
    G0_csr_data = (Real*)malloc(G0_nnz * sizeof(Real));
    fread(G0_rowptr, sizeof(Int), G0_row + 1, ptr);
    fread(G0_columns, sizeof(Int), G0_nnz, ptr);
    fread(G0_csr_data, sizeof(Real), G0_nnz, ptr);


    fread(&G1_row, sizeof(Int), 1, ptr);
    fread(&G1_col, sizeof(Int), 1, ptr);
    fread(&G1_nnz, sizeof(Int), 1, ptr);
    G1_rowptr   = (Int*)malloc((G1_row + 1) * sizeof(Int));
    G1_columns  = (Int*)malloc(G1_nnz * sizeof(Int));
    G1_csr_data = (Real*)malloc(G1_nnz * sizeof(Real));
    fread(G1_rowptr, sizeof(Int), G1_row + 1, ptr);
    fread(G1_columns, sizeof(Int), G1_nnz, ptr);
    fread(G1_csr_data, sizeof(Real), G1_nnz, ptr);


    fread(&G2_row, sizeof(Int), 1, ptr);
    fread(&G2_col, sizeof(Int), 1, ptr);
    fread(&G2_nnz, sizeof(Int), 1, ptr);
    G2_rowptr   = (Int*)malloc((G2_row + 1) * sizeof(Int));
    G2_columns  = (Int*)malloc(G2_nnz * sizeof(Int));
    G2_csr_data = (Real*)malloc(G2_nnz * sizeof(Real));
    fread(G2_rowptr, sizeof(Int), G2_row + 1, ptr);
    fread(G2_columns, sizeof(Int), G2_nnz, ptr);
    fread(G2_csr_data, sizeof(Real), G2_nnz, ptr);


    fread(&G3_row, sizeof(Int), 1, ptr);
    fread(&G3_col, sizeof(Int), 1, ptr);
    fread(&G3_nnz, sizeof(Int), 1, ptr);
    G3_rowptr   = (Int*)malloc((G3_row + 1) * sizeof(Int));
    G3_columns  = (Int*)malloc(G3_nnz * sizeof(Int));
    G3_csr_data = (Real*)malloc(G3_nnz * sizeof(Real));
    fread(G3_rowptr, sizeof(Int), G3_row + 1, ptr);
    fread(G3_columns, sizeof(Int), G3_nnz, ptr);
    fread(G3_csr_data, sizeof(Real), G3_nnz, ptr);

#ifdef VERBOSE
    std::cout << "---------------------" << std::endl;
    std::cout << "data_row    : " << data_row << std::endl;
    std::cout << "data_dim    : " << data_dim << std::endl;
    std::cout << "-" << std::endl;
    std::cout << "queries_row : " << queries_row << std::endl;
    std::cout << "queries_dim : " << queries_dim << std::endl;
    std::cout << "-" << std::endl;
    std::cout << "G0_row      : " << G0_row   << std::endl;
    std::cout << "G0_col      : " << G0_col   << std::endl;
    std::cout << "G0_nnz      : " << G0_nnz   << std::endl;
    std::cout << "-" << std::endl;
    std::cout << "G1_row      : " << G1_row   << std::endl;
    std::cout << "G1_col      : " << G1_col   << std::endl;
    std::cout << "G1_nnz      : " << G1_nnz   << std::endl;
    std::cout << "-" << std::endl;
    std::cout << "G2_row      : " << G2_row   << std::endl;
    std::cout << "G2_col      : " << G2_col   << std::endl;
    std::cout << "G2_nnz      : " << G2_nnz   << std::endl;
    std::cout << "-" << std::endl;
    std::cout << "G3_row      : " << G3_row   << std::endl;
    std::cout << "G3_col      : " << G3_col   << std::endl;
    std::cout << "G3_nnz      : " << G3_nnz   << std::endl;
    std::cout << "---------------------" << std::endl;
#endif

}

// --
// Helpers

inline Real get_dist(const Real* x, const Real* y, const Int d) {
    Real dist = 0;
    for(Int i = 0; i < d; i++)
        dist -= x[i] * y[i];
    return dist;
}

inline void greedy_walk(
    const Real* q,
    Int& v_curr,
    Real& d_curr,
    const Int* rowptr,
    const Int* columns
) {
    bool changed = true;
    while(changed) {
        changed = false;

        Int start = rowptr[v_curr];
        Int end   = rowptr[v_curr + 1];
        for(Int offset = start; offset < end; ++offset) {
            Int  v = columns[offset];
            Real d = get_dist(q, data + (v * data_dim), data_dim);

            if(d < d_curr) {
                d_curr  = d;
                v_curr  = v;
                changed = true;
            }
        }
    }
}

class prioritize_min {
    public:
        bool operator()(pair<Int, Real> &p1, pair<Int, Real> &p2) { return p1.second > p2.second; }
};

class reserved_minheap : public std::priority_queue<pair<Int, Real>, std::vector<pair<Int, Real>>, prioritize_min> {
public:
    reserved_minheap(size_t reserve_size) { this->c.reserve(reserve_size); }
};

class prioritize_max {
    public:
        bool operator()(pair<Int, Real> &p1, pair<Int, Real> &p2) { return p1.second < p2.second; }
};

class reserved_maxheap : public std::priority_queue<pair<Int, Real>, std::vector<pair<Int, Real>>, prioritize_max> {
public:
    reserved_maxheap(size_t reserve_size) { this->c.reserve(reserve_size); }
};

// --
// ipnsw

void ipnsw(Real* q, Int* s, bool* seen) {
    Int v_curr  = v_entry;
    Real d_curr = get_dist(q, data + (v_curr * data_dim), data_dim);

    greedy_walk(q, v_curr, d_curr, G3_rowptr, G3_columns);
    greedy_walk(q, v_curr, d_curr, G2_rowptr, G2_columns);
    greedy_walk(q, v_curr, d_curr, G1_rowptr, G1_columns);
    reserved_minheap candidates(1024);
    reserved_maxheap results(ef + 1);

    candidates.push(make_pair(v_curr, d_curr));
    results.push(make_pair(v_curr, d_curr));

    memset(seen, false, G0_row * sizeof(bool));
    seen[v_curr] = true;

    while(!candidates.empty()) {
        pair<Int, Real> best = candidates.top();
        candidates.pop();

        pair <Int, Real> worst = results.top();
        if(best.second > worst.second) break;

        Int start = G0_rowptr[best.first];
        Int end   = G0_rowptr[best.first + 1];

        for(Int offset = start; offset < end; offset++) {
            Int neib = G0_columns[offset];

            if(seen[neib]) continue;
            seen[neib] = true;

            Real d_neib = get_dist(q, data + (neib * data_dim), data_dim);

            if(((d_neib < worst.second) || results.size() < ef)) {
                candidates.push(make_pair(neib, d_neib));
                results.push(make_pair(neib, d_neib));

                while(results.size() > ef) results.pop();
                worst = results.top();
            }
        }
    }

    while(results.size() > n_results) results.pop();

    for(unsigned int i = 0; i < n_results; i++) {
        pair<Int, Real> result = results.top();
        results.pop();
        s[n_results - i - 1] = result.first;
    }
}

// --
// Run

void run_app(Int* scores) {
    int n_threads;
    #pragma omp parallel
    {
        n_threads = omp_get_num_threads();
    }
    
    bool* all_seen = (bool*)malloc(n_threads * G0_row * sizeof(bool));

    #pragma omp parallel for schedule(dynamic)
    for(Int i = 0; i < queries_row; i++) {
        const int tid = omp_get_thread_num();
        ipnsw(
            queries  + (i * queries_dim),
            scores   + (i * n_results),
            all_seen + (tid * G0_row)
        );
    }   
    
    free(all_seen);
}

int main(int n_args, char** argument_array) {
    // ---------------- INPUT ----------------
    
    if(n_args > 2) n_runs = (unsigned int)atoi(argument_array[2]);
    if(n_args > 3) ef     = (unsigned int)atoi(argument_array[3]);
    
    load_data(argument_array[1]);
    
    // ---------------- RUN ----------------
    auto t1 = high_resolution_clock::now();
    
    scores = (Int*)malloc(queries_row * n_results * sizeof(Int));
    
    for(unsigned int run = 0; run < n_runs; run++)
        run_app(scores);
    
    auto elapsed = high_resolution_clock::now() - t1;
    long long ms = duration_cast<microseconds>(elapsed).count();

    long long ms_avg = ms / n_runs;

    // ---------------- OUTPUT ----------------
    
    std::cout << 
        " > n_runs="     << n_runs      << 
        " | ef="         << ef          << 
        " | n_data="     << data_row    << 
        " | n_queries="  << queries_row << 
        " | ms="         << ms          <<
        " | ms_avg="     << ms_avg      << 
        " | throughput=" << 1e6 * ((Real)queries_row) / ms_avg << std::endl;
    
    // Write scores
    std::ofstream scores_file;
    scores_file.open("results/scores");

    for(Int i = 0; i < queries_row; i++) {
        for(Int j = 0; j < n_results; j++) {
            scores_file << scores[i * n_results + j] << " ";
        }
        scores_file << std::endl;
    }

    scores_file.close();

    // Write elapsed
    std::ofstream elapsed_file;
    elapsed_file.open("results/elapsed");
    elapsed_file << ms << std::endl;
    elapsed_file.close();

    return 0;
}
