#include <iostream>
#include <limits>
#include <cmath>
#include <assert.h>
#include <stdio.h>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

using namespace std;

struct PseudoJet
{
    double px;
    double py;
    double pz;
    double E;
    double diB;
    double phi;
    double rap;
};

const double pi = 3.141592653589793238462643383279502884197;
const double twopi = 6.283185307179586476925286766559005768394;
const double MaxRap = 1e5;
const double R = 0.6;
const double R2 = R * R;
const double invR2 = 1.0 / R2;
const double ptmin = 5.0;
const double dcut = ptmin * ptmin;
int const NUM_PARTICLES = 354;

__device__ void _set_jet(PseudoJet &jet)
{
    jet.diB = jet.px * jet.px + jet.py * jet.py;

    if (jet.diB == 0.0)
    {
        jet.phi = 0.0;
    }
    else
    {
        jet.phi = atan2(jet.py, jet.px);
    }
    if (jet.phi < 0.0)
    {
        jet.phi += twopi;
    }
    if (jet.phi >= twopi)
    {
        jet.phi -= twopi;
    } // can happen if phi=-|eps<1e-15|?
    if (jet.E == abs(jet.pz) && jet.diB == 0)
    {
        // Point has infinite rapidity -- convert that into a very large
        // number, but in such a way that different 0-pt momenta will have
        // different rapidities (so as to lift the degeneracy between
        // them) [this can be relevant at parton-level]
        double MaxRapHere = MaxRap + abs(jet.pz);
        if (jet.pz >= 0.0)
        {
            jet.rap = MaxRapHere;
        }
        else
        {
            jet.rap = -MaxRapHere;
        }
    }
    else
    {
        // get the rapidity in a way that's modestly insensitive to roundoff
        // error when things pz,E are large (actually the best we can do without
        // explicit knowledge of mass)
        double effective_m2 = max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - jet.diB); // force non tachyonic mass
        double E_plus_pz = jet.E + abs(jet.pz);                                        // the safer of p+, p-
        // p+/p- = (p+ p-) / (p-)^2 = (kt^2+m^2)/(p-)^2
        jet.rap = 0.5 * log((jet.diB + effective_m2) / (E_plus_pz * E_plus_pz));
        if (jet.pz > 0)
        {
            jet.rap = -jet.rap;
        }
    }
}

__device__ double plain_distance(PseudoJet &jet1, PseudoJet &jet2)
{
    double dphi = abs(jet1.phi - jet2.phi);
    if (dphi > pi)
    {
        dphi = twopi - dphi;
    }
    double drap = jet1.rap - jet2.rap;
    return (dphi * dphi + drap * drap);
}

__device__ double yij_distance(PseudoJet &jet1, PseudoJet &jet2)
{
    return min(jet1.diB, jet2.diB) *
           plain_distance(jet1, jet2) *
           invR2;
}

double plain_distance_h(PseudoJet &jet1, PseudoJet &jet2)
{
    double dphi = abs(jet1.phi - jet2.phi);
    if (dphi > pi)
    {
        dphi = twopi - dphi;
    }
    double drap = jet1.rap - jet2.rap;
    return (dphi * dphi + drap * drap);
}

double yij_distance_h(PseudoJet &jet1, PseudoJet &jet2)
{
    return min(jet1.diB, jet2.diB) *
           plain_distance_h(jet1, jet2) *
           invR2;
}

__global__ void set_jets(PseudoJet *jets)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    _set_jet(jets[tid]);
    // __syncthreads();
    // if(tid == 0)
    //     for(int i = 0; i < NUM_PARTICLES; i++)
    //         printf("%10.8f%10.8f%10.8f%10.8f%10.8f%10.8f%10.8f\n",
    //             jets[i].px,
    //             jets[i].py,
    //             jets[i].pz,
    //             jets[i].E,
    //             jets[i].diB,
    //             jets[i].phi,
    //             jets[i].rap
    //         );
}

__global__ void set_distances(PseudoJet *jets, double *distances,
                              int const num_particles)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i = tid / num_particles;
    int j = (tid % num_particles) - i;

    if (i == j)
    {
        distances[tid] = jets[i].diB;
    }
    else
    {
        distances[tid] = yij_distance(jets[i], jets[j]);
    }

    if (tid == 0)
    //     for (int tid = 0; tid < gridDim.x * blockDim.x; tid++)
    //     {
    //         int i = tid / NUM_PARTICLES;
    //         int j = (tid % NUM_PARTICLES) - 1;

    //         if (i == j)
    //         {
    //             // cout << endl
    //             //      << h_distances[tid];
    //             if (distances[tid] == jets[i].diB)
    //                 printf("\n0");
    //             else
    //                 printf("\n1");
    //         }
    //         else
    //         {
    //             // cout << endl
    //             //      << h_distances[tid];
    //             if (distances[tid] == yij_distance(jets[i], jets[j]))
    //                 printf(" 0");
    //             else
    //                 printf(" 1");
    //         }
    //     }
    for (int i = 0; i < gridDim.x * blockDim.x; i++)
        printf("%d %10.5f\n", i, distances[i]);
}

__global__ void reduction_min(PseudoJet *jets, double *distances,
                              int const num_particles)
{
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (threadIdx.x % (2 * s) == 0)
            if (threadIdx.x + s < num_particles)
                if (distances[threadIdx.x] >
                    distances[threadIdx.x + s])
                {
                    distances[threadIdx.x] =
                        distances[threadIdx.x + s];
                }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        printf("%15.8e\n", distances[blockDim.x * blockIdx.x]);
}

int main()
{
    int d_id;
    cudaDeviceProp d_prop;

    cudaChooseDevice(&d_id, &d_prop);
    cout << "device id is " << d_id << endl;
    cudaSetDevice(d_id);

    PseudoJet *h_jets = 0;
    h_jets = (PseudoJet *)malloc(NUM_PARTICLES * sizeof(PseudoJet));

    double *h_mini = 0;
    h_mini = (double *)malloc(sizeof(double));
    *h_mini = numeric_limits<double>::max();

    int i;
    for (i = 0; i < NUM_PARTICLES; i++)
    {
        cin >> h_jets[i].px >> h_jets[i].py >> h_jets[i].pz >> h_jets[i].E;
    }

    PseudoJet *d_jets = 0;
    cudaMalloc((void **)&d_jets, NUM_PARTICLES * sizeof(PseudoJet));
    cudaMemcpy(d_jets, h_jets, NUM_PARTICLES * sizeof(PseudoJet), cudaMemcpyHostToDevice);

    double *d_distances = 0;
    cudaMalloc((void **)&d_distances,
               (NUM_PARTICLES * (NUM_PARTICLES + 1) / 2) * sizeof(double));

    double *d_mini = 0;
    cudaMalloc((void **)&d_mini, sizeof(double));
    cudaMemcpy(d_mini, h_mini, sizeof(double), cudaMemcpyHostToDevice);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy calls1");

    int num_threads = 354;
    int num_blocks = (NUM_PARTICLES + num_threads) / num_threads + 1;
    //std::cout << "blocks = " << num_blocks;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    set_jets<<<num_blocks, num_threads>>>(d_jets);

    num_threads = (NUM_PARTICLES * (NUM_PARTICLES + 1) / 2);
    num_blocks = (num_threads / 1024) + 1;
    cout << num_threads << " " << num_blocks << endl;
    set_distances<<<num_blocks, 1024>>>(d_jets, d_distances,
                                        NUM_PARTICLES);
    reduction_min<<<num_blocks, 1024>>>(d_jets, d_distances,
                                        NUM_PARTICLES);
    cudaEventRecord(stop);

    // Check for any CUDA errors
    checkCUDAError("kernal launch");
    cudaMemcpy(h_jets, d_jets,
               NUM_PARTICLES * sizeof(PseudoJet),
               cudaMemcpyDeviceToHost);

    double *h_distances = 0;
    h_distances = (double *)malloc(num_threads * sizeof(double));
    cudaMemcpy(h_distances, d_distances, num_threads * sizeof(double),
               cudaMemcpyDeviceToHost);

    // for (int tid = 0; tid < num_threads; tid++)
    // {
    //     int i = tid / NUM_PARTICLES;
    //     int j = (tid % NUM_PARTICLES) - 1;

    //     cout << h_distances[tid] << endl;
    //     // if (i == j)
    //     // {
    //     //     cout << endl
    //     //          << h_distances[tid];
    //     //     // if (h_distances[tid] == h_jets[i].diB)
    //     //     //     cout << "\n0";
    //     //     // else
    //     //     //     cout << "\n1";
    //     // }
    //     // else
    //     // {
    //     //     cout << endl
    //     //          << h_distances[tid];
    //     //     // if (h_distances[tid] == yij_distance_h(h_jets[i], h_jets[j]))
    //     //     //     cout << " 0";
    //     //     // else
    //     //     //     cout << " 1";
    //     // }
    // }

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy2 calls");

    cudaMemcpy(h_mini, d_mini, sizeof(double), cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy3 calls");

    //    cout << "d_mini = " << *h_mini << std::endl;
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time = %10.8f\n", milliseconds);

    // free device memory
    cudaFree(d_jets);
    cudaFree(d_mini);
    cudaFree(d_distances);

    // free host memory
    free(h_jets);
    free(h_mini);
    free(h_distances);

    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
