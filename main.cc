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

int grid_size(double min_rap, double max_rap, double min_phi, double max_phi, double r, int n) {
  r = (2 * M_PI) / (int)((2 * M_PI) / r);
  return (int)((max_rap - min_rap) / r) * (int)((max_phi - min_phi) / r) * n;
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
int read_n_events(std::istream& input, std::vector<PseudoJet>& particles, int combine) {
  // clear the output buffer
  particles.clear();

  int events = 0;
  while ((combine == 0 or events < combine) and read_next_event(input, particles))
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
  int events_per_kernel = 1;

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

        // --stream-count, -sc
        if (std::strcmp(argv[i], "--stream-count") == 0 or std::strcmp(argv[i], "-sc") == 0) {
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

        // --evevnts-per-kernel, -e
        if (std::strcmp(argv[i], "--events-per-kernel") == 0 or std::strcmp(argv[i], "-e") == 0) {
      ++i;
      if (i >= argc) {
        // error
        std::cerr << "Missing argument to option " << argv[i - 1] << std::endl;
        return 1;
      }
      char* stop;
      auto arg = std::strtol(argv[i], &stop, 0);
      if (stop != argv[i] and arg >= 0) {
        events_per_kernel = arg;
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
    const size_t max_event_size = 3000;
    const size_t particles_per_kernel = max_event_size * events_per_kernel;
    const size_t total_particles = particles_per_kernel * stream_count;
    const size_t size_of_one_grid = grid_size(-10., +10., 0, 2 * M_PI, r, max_event_size);

    const size_t max_global_memory_needed =
        sizeof(PseudoJet) * total_particles + sizeof(int) * size_of_one_grid * events_per_kernel * stream_count +
        sizeof(int) * total_particles + sizeof(PseudoJetExt) * total_particles + sizeof(Dist) * total_particles;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (max_global_memory_needed > prop.totalGlobalMem) {
      std::cerr << "The GPU does not have enough memory!\n" << std::endl;
      return 1;
    }

    std::vector<PseudoJet>* events = new std::vector<PseudoJet>[stream_count];
    std::vector<PseudoJet>* jets = new std::vector<PseudoJet>[stream_count];
    std::vector<int>* events_sizes = new std::vector<int>[stream_count];

    PseudoJet* h_events;
    PseudoJet* h_jets;
    cudaCheck(cudaMallocHost(&h_events, sizeof(PseudoJet) * total_particles));
    cudaCheck(cudaMallocHost(&h_jets, sizeof(PseudoJet) * total_particles));

    // allocate GPU memory for the input particles
    PseudoJet* d_particles;
    cudaCheck(cudaMalloc(&d_particles, sizeof(PseudoJet) * total_particles));

    int* d_grid;
    cudaCheck(cudaMalloc(&d_grid, sizeof(int) * size_of_one_grid * events_per_kernel * stream_count));

    int* d_event_sizes;
    cudaCheck(cudaMalloc(&d_event_sizes, sizeof(int) * total_particles));

    PseudoJetExt* d_pseudojets;
    cudaCheck(cudaMalloc(&d_pseudojets, sizeof(PseudoJetExt) * total_particles));

    Dist* d_min_dists_ptr;
    cudaCheck(cudaMalloc(&d_min_dists_ptr, sizeof(Dist) * total_particles));

    cudaEvent_t* starts = new cudaEvent_t[stream_count + 1];
    cudaEvent_t* ends = new cudaEvent_t[stream_count + 1];

    bool* first_run = new bool[stream_count];
    cudaStream_t* stream = new cudaStream_t[stream_count];
    for (int i = 0; i < stream_count; ++i) {
      cudaCheck(cudaStreamCreate(&stream[i]));
      cudaCheck(cudaEventCreate(&starts[i]));
      cudaCheck(cudaEventCreate(&ends[i]));
      first_run[i] = true;
    }

    cudaCheck(cudaEventCreate(&starts[stream_count]));
    cudaCheck(cudaEventCreate(&ends[stream_count]));
    cudaCheck(cudaEventRecord(starts[stream_count]));
    int stream_id = 0;
    while (read_n_events(filename.empty() ? std::cin : input, events[stream_id], combine)) {
      std::cout << "found " << events[stream_id].size() << " particles\n";
      events_sizes[stream_id].push_back(events[stream_id].size());
      std::copy(events[stream_id].begin(), events[stream_id].end(), &h_events[stream_id * particles_per_kernel]);
      if (not first_run[stream_id]) {
        cudaCheck(cudaEventSynchronize(ends[stream_id]));
        jets[stream_id].insert(jets[stream_id].begin(),
                               &h_jets[stream_id * particles_per_kernel],
                               &h_jets[stream_id * particles_per_kernel + jets[stream_id].size()]);
        float milliseconds;
        cudaCheck(cudaEventElapsedTime(&milliseconds, starts[stream_id], ends[stream_id]));

        // remove the unused elements and the jets with pT < pTmin
        auto last = std::remove_if(jets[stream_id].begin(), jets[stream_id].end(), [ptmin](auto const& jet) {
          return (not jet.isJet) or (jet.px * jet.px + jet.py * jet.py < ptmin * ptmin);
        });
        jets[stream_id].erase(last, jets[stream_id].end());

        if (ptmin > 0.) {
          std::cout << "found " << jets[stream_id].size() << " jets above " << ptmin << " GeV in " << milliseconds
                    << "ms\n";
        } else {
          std::cout << "found " << jets[stream_id].size() << std::endl;
        }
        jets[stream_id].clear();
      }

      first_run[stream_id] = false;

      cudaCheck(cudaEventRecord(starts[stream_id]));

      // copy the input to the GPU
      cudaCheck(cudaMemcpyAsync(&d_particles[stream_id * particles_per_kernel],
                                &h_events[stream_id * particles_per_kernel],
                                sizeof(PseudoJet) * events[stream_id].size(),
                                cudaMemcpyDefault,
                                stream[stream_id]));

      // run the clustering algorithm
      cluster(&d_particles[stream_id * particles_per_kernel],
              events[stream_id].size(),
              algo,
              r,
              stream[stream_id],
              &d_grid[stream_id * size_of_one_grid * events_per_kernel],
              &d_pseudojets[stream_id * particles_per_kernel],
              &d_min_dists_ptr[stream_id * particles_per_kernel]);

      // copy the clustered jets back to the CPU
      jets[stream_id].resize(events[stream_id].size());

      cudaCheck(cudaMemcpyAsync(&h_jets[stream_id * particles_per_kernel],
                                &d_particles[stream_id * particles_per_kernel],
                                sizeof(PseudoJet) * jets[stream_id].size(),
                                cudaMemcpyDefault,
                                stream[stream_id]));

      cudaCheck(cudaEventRecord(ends[stream_id]));

      stream_id = (stream_id + 1) % stream_count;
    }

    for (int i = 0; i < stream_count; ++i) {
      if (jets[stream_id].size() > 0) {
        cudaCheck(cudaEventSynchronize(ends[stream_id]));
        jets[stream_id].insert(jets[stream_id].begin(),
                               &h_jets[stream_id * particles_per_kernel],
                               &h_jets[stream_id * particles_per_kernel + jets[stream_id].size()]);
        float milliseconds;
        cudaCheck(cudaEventElapsedTime(&milliseconds, starts[stream_id], ends[stream_id]));

        // remove the unused elements and the jets with pT < pTmin
        auto last = std::remove_if(jets[stream_id].begin(), jets[stream_id].end(), [ptmin](auto const& jet) {
          return (not jet.isJet) or (jet.px * jet.px + jet.py * jet.py < ptmin * ptmin);
        });
        jets[stream_id].erase(last, jets[stream_id].end());

        if (ptmin > 0.) {
          std::cout << "found " << jets[stream_id].size() << " jets above " << ptmin << " GeV in " << milliseconds
                    << "ms\n";
        } else {
          std::cout << "found " << jets[stream_id].size() << std::endl;
        }
      }
      stream_id = (stream_id + 1) % stream_count;
    }

    cudaCheck(cudaEventRecord(ends[stream_count]));
    cudaCheck(cudaEventSynchronize(ends[stream_count]));
    float milliseconds;
    cudaCheck(cudaEventElapsedTime(&milliseconds, starts[stream_count], ends[stream_count]));

    std::cout << "Total prcessing time: " << milliseconds << "ms\n";

    cudaCheck(cudaFree(d_particles));
    cudaCheck(cudaFree(d_grid));
    cudaCheck(cudaFree(d_pseudojets));
    cudaCheck(cudaFree(d_min_dists_ptr));
    for (int i = 0; i < stream_count; ++i) {
      cudaStreamDestroy(stream[i]);
    }

    delete[] stream;
    delete[] events;
    std::cout << grid_size(-10., +10., 0, 2 * M_PI, r, max_event_size) << std::endl;
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
      PseudoJet* d_particles;
      cudaCheck(cudaMalloc(&d_particles, sizeof(PseudoJet) * particles.size()));

      cudaEvent_t start, stop;
      cudaCheck(cudaEventCreate(&start));
      cudaCheck(cudaEventCreate(&stop));

      double sum = 0.;
      double sum2 = 0.;
      for (int step = 0; repetitions == 0 or step < repetitions; ++step) {
        cudaCheck(cudaEventRecord(start));

        // copy the input to the GPU
        cudaCheck(cudaMemcpy(d_particles, particles.data(), sizeof(PseudoJet) * particles.size(), cudaMemcpyDefault));

        // run the clustering algorithm and measure its running time
        //cluster(d_particles, particles.size(), algo, r);

        // copy the clustered jets back to the CPU
        jets.resize(particles.size());
        cudaCheck(cudaMemcpy(jets.data(), d_particles, sizeof(PseudoJet) * jets.size(), cudaMemcpyDefault));

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
      cudaCheck(cudaFree(d_particles));

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
