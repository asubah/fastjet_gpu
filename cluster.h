#ifndef cluster_h
#define cluster_h

enum class Algorithm { Kt = 1, CambridgeAachen = 0, AntiKt = -1 };

void cluster(PseudoJet *particles, int *events_sizes, int size, Algorithm algo, double r, int max_event_size, cudaStream_t stream, int *grid_ptr, PseudoJetExt *pseudojets, Dist *d_min_dists_ptr);

#endif  // cluster_h
