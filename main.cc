#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "PseudoJet.h"
#include "cluster.h"
#include "cudaCheck.h"


void initialise() {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Running on CUDA device " << prop.name << std::endl;
}


bool read_next_event(std::istream& input, std::vector<PseudoJet> & particles) {
  // clear the output buffer
  particles.clear();

  // clear the input status flags
  input.clear();

  // skip comments and empty lines
  while (input.peek() == '#' or input.peek() == '\n') {
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  // read the input one line at a time
  int i = 0; 
  std::string buffer;
  while (std::getline(input, buffer).good()) {
    std::istringstream line(buffer);

    // read the four elements
    double px, py, pz, E;
    line >> px >> py >> pz >> E;
    //std::cout << "reading: " << px << ", " << py << ", " << pz << ", " << E << std::endl;

    if (line.fail()) {
      //std::cout << "no more particles" << std::endl;
      // check for a comment or empty line
      if (not buffer.empty() and buffer[0] != '#') {
        throw std::runtime_error("Error while parsing particles:\n" + buffer);
      }
      break;
    }

    //std::cout << "found a particle" << std::endl;
    particles.push_back({i++, false, px, py, pz, E});
  }

  // return false if there was no event to read
  return (not particles.empty());
}

void print_jets(std::vector<PseudoJet> const& jets) {
    std::cout << std::fixed << std::setprecision(6);
    for (auto const& jet: jets) {
      std::cout << std::setw(16) << jet.px << std::setw(16) << jet.py << std::setw(16) << jet.pz << std::setw(16) << jet.E << std::endl;
    }
    std::cout << std::endl;
}


int main(int argc, const char* argv[]) {
  double ptmin = 1.0;   // GeV
  bool sort = true;
  int repetitions = 1;

  for (unsigned int i = 1; i < argc; ++i) {
    // --ptmin, -p
    if (std::strcmp(argv[i], "--ptmin") == 0 or std::strcmp(argv[i], "-p") == 0) {
      ++i;
      if (i >= argc) {
        // error
        return 1;
      }
      char* stop;
      auto arg = std::strtod(argv[i], &stop);
      if (stop != argv[i] and arg >= 0.) {
        ptmin = arg;
      } else {
        // error
        return 1;
      }
    } else

    // --repeat, -r
    if (std::strcmp(argv[i], "--repeat") == 0 or std::strcmp(argv[i], "-r") == 0) {
      ++i;
      if (i >= argc) {
        // error
        return 1;
      }
      char* stop;
      auto arg = std::strtol(argv[i], &stop, 0);
      if (stop != argv[i] and arg >= 0) {
        repetitions = arg;
      } else {
        // error
        return 1;
      }
    } else

    // --sort, -s
    if (std::strcmp(argv[i], "--sort") == 0 or std::strcmp(argv[i], "-s") == 0) {
      sort = true;
    } else

    // unknown option
    {
      // error
      return 1;
    }
  }


  std::vector<PseudoJet> particles;
  std::vector<PseudoJet> jets;

  while (read_next_event(std::cin, particles)) {
    std::cout << "found " << particles.size() << " particles" << std::endl;

    // allocate GPU memory for the input particles
    PseudoJet * particles_d;
    cudaCheck(cudaMalloc(&particles_d, sizeof(PseudoJet) * particles.size()));

    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    double sum = 0.;
    double sum2 = 0.;
    for (int step = 0; repetitions == 0 or step < repetitions; ++step) {
      // copy the input to the GPU
      cudaCheck(cudaMemcpy(particles_d, particles.data(), sizeof(PseudoJet) * particles.size(), cudaMemcpyDefault));

      // run the clustering algorithm and measure its running time
      cudaCheck(cudaEventRecord(start));
      cluster(particles_d, particles.size());
      cudaCheck(cudaEventRecord(stop));
      cudaCheck(cudaEventSynchronize(stop));

      float milliseconds;
      cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
      sum += milliseconds;
      sum2 += milliseconds * milliseconds;

      // copy the clustered jets back to the CPU
      jets.resize(particles.size());
      cudaCheck(cudaMemcpy(jets.data(), particles_d, sizeof(PseudoJet) * jets.size(), cudaMemcpyDefault));

      // remove the unused elements and the jets with pT < pTmin
      auto last = std::remove_if(jets.begin(), jets.end(), [ptmin](auto const& jet){
        return (not jet.isJet) or (jet.px * jet.px + jet.py * jet.py < ptmin * ptmin);
      });
      jets.erase(last, jets.end());

      if (ptmin > 0.) {
        std::cout << "found " << jets.size() << " jets above " << ptmin << " GeV in " << milliseconds << " ms" << std::endl;
      } else {
        std::cout << "found " << jets.size() << " jets in " << milliseconds << " ms" << std::endl;
      }
    
      // optionally, sort the jets by decreasing pT
      if (sort) {
        std::sort(jets.begin(), jets.end(), [](auto const& a, auto const& b) {
          return (a.px * a.px + a.py * a.py < b.px * b.px + b.py * b.py);
        });
      }
    }

    // free GPU memory
    cudaCheck(cudaFree(particles_d));

    print_jets(jets);

    double mean  = sum / repetitions;
    double sigma = std::sqrt((sum2 - sum * sum / repetitions) / (repetitions - 1));
    int precision = std::max((int) -std::log10(sigma / 2.) + 1, 0);
    std::cout << std::defaultfloat;
    std::cout << "clustered " << particles.size() << " into " << jets.size() << " jets above " << ptmin << " GeV";
    precision = std::cout.precision(precision);
    std::cout << std::fixed;
    std::cout << " in " << mean << " +/- " << sigma << " ms" << std::endl;
    std::cout.precision(precision);
    std::cout << std::defaultfloat;
  }

  return 0;
}