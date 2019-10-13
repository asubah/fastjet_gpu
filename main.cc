#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
  int value;
  cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  std::cout << "  - maximum shared memory per block: " << value / 1024 << " kB" << std::endl;

  cudaCheck(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024));
  size_t size;
  cudaCheck(cudaDeviceGetLimit(&size, cudaLimitPrintfFifoSize));
  std::cout << "  - kernel printf buffer size:       " << size / 1024 << " kB" << std::endl;
}

int grid_memory_size(double min_rap, double max_rap, double min_phi, double max_phi, double r, int n) {
  r = (2 * M_PI) / (int)((2 * M_PI) / r);
  return sizeof(int) * (int)((max_rap - min_rap) / r) * (int)((max_phi - min_phi) / r) * n;
}

bool read_next_event(std::istream& input, std::vector<PseudoJet>& particles) {
  // clear the input status flags
  input.clear();

  // skip comments and empty lines
  while (input.peek() == '#' or input.peek() == '\n') {
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  // read the input one line at a time
  int i = particles.size();
  bool found = false;
  std::string buffer;
  while (std::getline(input, buffer).good()) {
    std::istringstream line(buffer);

    // read the four elements
    double px, py, pz, E;
    line >> px >> py >> pz >> E;
    //std::cout << "reading: " << px << ", " << py << ", " << pz << ", " << E << std::endl;

    if (line.fail()) {
      // check for a comment or empty line
      if (not buffer.empty() and buffer[0] != '#') {
        throw std::runtime_error("Error while parsing particles:\n" + buffer);
      }
      break;
    }

    //std::cout << "found a particle" << std::endl;
    particles.push_back({i++, false, px, py, pz, E});
    found = true;
  }

  // return false if there was no event to read
  return (found);
}

/* Read the next N events from the input stream, and returns the number of events actually read.
 *
 * Pass 0 to read all events in the stream.
 */
int read_n_events(std::istream& input, std::vector<PseudoJet>& particles, int n) {
  // clear the output buffer
  particles.clear();

  int events = 0;
  while ((n == 0 or events < n) and read_next_event(input, particles))
    ++events;

  return events;
}

void print_jets(std::vector<PseudoJet> const& jets, bool cartesian = false) {
  std::cout << std::fixed << std::setprecision(8);
  int i = 0;
  for (auto const& jet : jets) {
    if (cartesian) {
      // print px, py, pz, E
      std::cout << std::setw(5) << i++ << std::setw(16) << jet.px << std::setw(16) << jet.py << std::setw(16) << jet.pz
                << std::setw(16) << jet.E << std::endl;
    } else {
      // print eta, phi, pT
      double pT = std::hypot(jet.px, jet.py);
      double phi = atan2(jet.py, jet.px);
      while (phi > 2 * M_PI) {
        phi -= 2 * M_PI;
      }
      while (phi < 0.) {
        phi += 2 * M_PI;
      }
      double effective_m2 = std::max(0.0, (jet.E + jet.pz) * (jet.E - jet.pz) - pT * pT);
      double E_plus_pz = jet.E + std::abs(jet.pz);
      double eta = 0.5 * std::log((pT * pT + effective_m2) / (E_plus_pz * E_plus_pz));
      if (jet.pz > 0) {
        eta = -eta;
      }
      std::cout << std::setw(5) << i++ << std::setw(16) << eta << std::setw(16) << phi << std::setw(16) << pT
                << std::endl;
    }
  }
  std::cout << std::endl;
}

int main(int argc, const char* argv[]) {
  double ptmin = 0.0;              // GeV
  double r = 1.0;                  // clustering radius
  Algorithm algo = Algorithm::Kt;  // clustering algorithm
  bool sort = true;
  bool cartesian = false;
  int repetitions = 1;
  int combine = 1;
  std::string filename;  // read data from file instead of standard input
  bool output_csv = false;
  int stream_count = 1;

  for (unsigned int i = 1; i < argc; ++i) {
    // --ptmin, -p
    if (std::strcmp(argv[i], "--ptmin") == 0 or std::strcmp(argv[i], "-p") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtod(argv[i], &stop);
      if (stop != argv[i] and arg >= 0.) {
        ptmin = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

        // -r, -R
        if (std::strcmp(argv[i], "-r") == 0 or std::strcmp(argv[i], "-R") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtod(argv[i], &stop);
      if (stop != argv[i] and arg >= 0) {
        r = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

        // --repeat, -repeat
        if (std::strcmp(argv[i], "--repeat") == 0 or std::strcmp(argv[i], "-repeat") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtol(argv[i], &stop, 0);
      if (stop != argv[i] and arg >= 0) {
        repetitions = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

        // --sort, -s
        if (std::strcmp(argv[i], "--sort") == 0 or std::strcmp(argv[i], "-s") == 0) {
      sort = true;
    } else

        // --cartesian
        if (std::strcmp(argv[i], "--cartesian") == 0) {
      cartesian = true;
    } else

        // --polar
        if (std::strcmp(argv[i], "--polar") == 0) {
      cartesian = false;
    } else

        // --kt, -kt
        if (std::strcmp(argv[i], "--kt") == 0 or std::strcmp(argv[i], "-kt") == 0) {
      algo = Algorithm::Kt;
    } else

        // --anti-kt, -antikt
        if (std::strcmp(argv[i], "--anti-kt") == 0 or std::strcmp(argv[i], "-antikt") == 0) {
      algo = Algorithm::AntiKt;
    } else

        // --cambridge-aachen, -cam
        if (std::strcmp(argv[i], "--cambridge-aachen") == 0 or std::strcmp(argv[i], "-cam") == 0) {
      algo = Algorithm::CambridgeAachen;
    } else

        // --file, -f
        if (std::strcmp(argv[i], "--file") == 0 or std::strcmp(argv[i], "-f") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      filename = argv[i];
    } else

        // --combine, -combine
        if (std::strcmp(argv[i], "--combine") == 0 or std::strcmp(argv[i], "-combine") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtol(argv[i], &stop, 0);
      if (stop != argv[i] and arg >= 0) {
        combine = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

        // --csv, -csv
        if (std::strcmp(argv[i], "--csv") == 0 or std::strcmp(argv[i], "-csv") == 0) {
      output_csv = true;
    } else

        // --stream_count, -sc
        if (std::strcmp(argv[i], "--stream_count") == 0 or std::strcmp(argv[i], "-sc") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtol(argv[i], &stop, 0);
      if (stop != argv[i] and arg >= 0) {
        stream_count = arg;
      } else {
        // error
        std::cerr << "Error while parsing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
    } else

    // unknown option
    {
      std::cerr << "Unrecognized option " << argv[i] << std::endl;
      return 1;
    }
  }

  // initialise the GPU
  initialise();

  // open an input file
  std::ifstream input(filename, std::ios_base::in);

  if (stream_count > 1) {
    std::cout << "Stream count is " << stream_count << "\n";
    const int MAX_EVENT_SIZE = 3000;

    std::vector<PseudoJet>* events = new std::vector<PseudoJet>[stream_count];
    std::vector<PseudoJet>* jets = new std::vector<PseudoJet>[stream_count];

    // allocate GPU memory for the input particles
    PseudoJet** particles_d = new PseudoJet*[stream_count];
    int** d_grid = new int*[stream_count];
    PseudoJetExt** pseudojets = new PseudoJetExt*[stream_count];
    Dist** d_min_dists_ptr = new Dist*[stream_count];
    
    cudaEvent_t *starts = new cudaEvent_t[stream_count];
    cudaEvent_t *ends = new cudaEvent_t[stream_count];

    int stream_id = 0;
    cudaStream_t* stream = new cudaStream_t[stream_count];
    for (int i = 0; i < stream_count; ++i) {
      std::cout << "malloc " << i << "\n";
      cudaCheck(cudaMalloc(&particles_d[i], sizeof(PseudoJet) * MAX_EVENT_SIZE));
      cudaCheck(cudaMalloc(&d_grid[i], grid_memory_size(-10., +10., 0, 2 * M_PI, r, MAX_EVENT_SIZE)));
      cudaCheck(cudaMalloc(&pseudojets[i], sizeof(PseudoJetExt) * MAX_EVENT_SIZE));
      cudaCheck(cudaMalloc(&d_min_dists_ptr[i], sizeof(Dist) * MAX_EVENT_SIZE));
      cudaCheck(cudaStreamCreate(&stream[i]));
      cudaCheck(cudaEventCreate(&starts[i]));
      cudaCheck(cudaEventCreate(&ends[i]));
    }

    while (read_n_events(filename.empty() ? std::cin : input, events[stream_id], combine)) {
      std::cout << "found " << events[stream_id].size() << " particles\n";
      
      cudaCheck(cudaEventRecord(starts[stream_id]));

      // copy the input to the GPU
      cudaCheck(cudaMemcpyAsync(particles_d[stream_id],
                                events[stream_id].data(),
                                sizeof(PseudoJet) * events[stream_id].size(),
                                cudaMemcpyDefault,
                                stream[stream_id]));
      // run the clustering algorithm
      cluster(particles_d[stream_id],
              events[stream_id].size(),
              algo,
              r,
              stream[stream_id],
              d_grid[stream_id],
              pseudojets[stream_id],
              d_min_dists_ptr[stream_id]);

      // copy the clustered jets back to the CPU
      jets[stream_id].resize(events[stream_id].size());
      cudaCheck(cudaMemcpyAsync(jets[stream_id].data(),
                                particles_d[stream_id],
                                sizeof(PseudoJet) * jets[stream_id].size(),
                                cudaMemcpyDefault,
                                stream[stream_id]));

      cudaCheck(cudaEventRecord(ends[stream_id]));      
      cudaCheck(cudaEventSynchronize(ends[stream_id]));

      if (stream_id + 1 == stream_count) {
        for (int i = 0; i < stream_count; ++i) {
          float milliseconds;
          cudaCheck(cudaEventElapsedTime(&milliseconds, starts[i], ends[i]));

          // remove the unused elements and the jets with pT < pTmin
          auto last = std::remove_if(jets[i].begin(), jets[i].end(), [ptmin](auto const& jet) {
            return (not jet.isJet) or (jet.px * jet.px + jet.py * jet.py < ptmin * ptmin);
          });
          jets[i].erase(last, jets[i].end());

          if (ptmin > 0.) {
            std::cout << "found " << jets[i].size() << " jets above " << ptmin << " GeV in " << milliseconds << "ms\n";
          } else {
            std::cout << "found " << jets[i].size() << std::endl;
          }
        }
      }

      stream_id = (stream_id + 1) % stream_count;
    }

    for (int i = 0; i < stream_count; ++i) {
      cudaStreamDestroy(stream[i]);

      cudaCheck(cudaFree(pseudojets[i]));
      cudaCheck(cudaFree(d_grid[i]));
      cudaCheck(cudaFree(d_min_dists_ptr[i]));
      cudaCheck(cudaFree(particles_d[i]));
    }

    delete[] stream;
    delete[] events;
    std::cout << grid_memory_size(-10., +10., 0, 2 * M_PI, r, MAX_EVENT_SIZE) << std::endl;
  } else {
    std::vector<PseudoJet> particles;
    std::vector<PseudoJet> jets;

    int events;
    while (events = read_n_events(filename.empty() ? std::cin : input, particles, combine)) {
      if (not output_csv) {
        std::cout << "found " << particles.size() << " particles";
        if (combine != 1)
          std::cout << " in " << events << (events == 1 ? " event" : " events");
        std::cout << std::endl;
      }

      // allocate GPU memory for the input particles
      PseudoJet* particles_d;
      cudaCheck(cudaMalloc(&particles_d, sizeof(PseudoJet) * particles.size()));

      cudaEvent_t start, stop;
      cudaCheck(cudaEventCreate(&start));
      cudaCheck(cudaEventCreate(&stop));

      double sum = 0.;
      double sum2 = 0.;
      for (int step = 0; repetitions == 0 or step < repetitions; ++step) {
        cudaCheck(cudaEventRecord(start));

        // copy the input to the GPU
        cudaCheck(cudaMemcpy(particles_d, particles.data(), sizeof(PseudoJet) * particles.size(), cudaMemcpyDefault));

        // run the clustering algorithm and measure its running time
        //cluster(particles_d, particles.size(), algo, r);

        // copy the clustered jets back to the CPU
        jets.resize(particles.size());
        cudaCheck(cudaMemcpy(jets.data(), particles_d, sizeof(PseudoJet) * jets.size(), cudaMemcpyDefault));

        cudaCheck(cudaEventRecord(stop));
        cudaCheck(cudaEventSynchronize(stop));

        float milliseconds;
        cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        sum += milliseconds;
        sum2 += milliseconds * milliseconds;

        // remove the unused elements and the jets with pT < pTmin
        auto last = std::remove_if(jets.begin(), jets.end(), [ptmin](auto const& jet) {
          return (not jet.isJet) or (jet.px * jet.px + jet.py * jet.py < ptmin * ptmin);
        });
        jets.erase(last, jets.end());

        if (not output_csv) {
          if (ptmin > 0.) {
            std::cout << "found " << jets.size() << " jets above " << ptmin << " GeV in " << milliseconds << " ms"
                      << std::endl;
          } else {
            std::cout << "found " << jets.size() << " jets in " << milliseconds << " ms" << std::endl;
          }

          // optionally, sort the jets by decreasing pT
          if (sort) {
            std::sort(jets.begin(), jets.end(), [](auto const& a, auto const& b) {
              return (a.px * a.px + a.py * a.py > b.px * b.px + b.py * b.py);
            });
          }
        }
      }

      // free GPU memory
      cudaCheck(cudaFree(particles_d));

      if (not output_csv)
        print_jets(jets, cartesian);

      std::cout << std::defaultfloat;

      if (not output_csv) {
        std::cout << "clustered " << particles.size() << " particles into " << jets.size() << " jets above " << ptmin
                  << " GeV";
      } else {
        std::cout << particles.size() << ',' << jets.size() << ',';
      }

      std::cout << std::fixed;
      double mean = sum / repetitions;
      int precision;
      if (repetitions > 1) {
        double sigma = std::sqrt((sum2 - sum * sum / repetitions) / (repetitions - 1));
        precision = std::max((int)-std::log10(sigma / 2.) + 1, 0);
        precision = std::cout.precision(precision);
        if (not output_csv) {
          std::cout << " in " << mean << " +/- " << sigma << " ms" << std::endl;
        } else {
          std::cout << mean << ',' << sigma << std::endl;
        }
      } else {
        precision = std::cout.precision(1);
        if (not output_csv) {
          std::cout << " in " << mean << " ms" << std::endl;
        } else {
          std::cout << mean << std::endl;
        }
      }
      std::cout.precision(precision);
      std::cout << std::defaultfloat;
    }
  }

  return 0;
}
