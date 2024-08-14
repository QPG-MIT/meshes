#include <math.h>
#include <stdio.h>
#include <string.h>

#define LOOP(k, N, EXPR) for (int k = 0; k < N; k++) {EXPR}
#define RMALLOC (real *) malloc

template <class real>
void split(real *dest_r, real *dest_i, real *src, int ct)
{
    int j = 0;
    for (int i = 0; i < ct; i++)
    {
        dest_r[i] = src[j++]; dest_i[i] = src[j++];
    }
}

template <class real>
void merge(real *dest, real *src_r, real *src_i, int ct)
{
    int j = 0;
    for (int i = 0; i < ct; i++)
    {
        dest[j++] = src_r[i]; dest[j++] = src_i[i];
    }
}

template <class real>
void permute(int N, int B, int dl, bool diff, int *perms, real *v_r, real *v_i, real *dv_r, real *dv_i, real *temp)
{
    for (int k = 0; k < N; k++) {memcpy(temp + B*k, v_r  + B*perms[k], dl*sizeof(real));}
    for (int k = 0; k < N; k++) {memcpy(v_r  + B*k, temp + B*k,        dl*sizeof(real));}
    for (int k = 0; k < N; k++) {memcpy(temp + B*k, v_i  + B*perms[k], dl*sizeof(real));}
    for (int k = 0; k < N; k++) {memcpy(v_i  + B*k, temp + B*k,        dl*sizeof(real));}
    if (diff)
    {
        for (int k = 0; k < N; k++) {memcpy(temp + B*k, dv_r + B*perms[k], dl*sizeof(real));}
        for (int k = 0; k < N; k++) {memcpy(dv_r + B*k, temp + B*k,        dl*sizeof(real));}
        for (int k = 0; k < N; k++) {memcpy(temp + B*k, dv_i + B*perms[k], dl*sizeof(real));}
        for (int k = 0; k < N; k++) {memcpy(dv_i + B*k, temp + B*k,        dl*sizeof(real));}
    }
}


#define NUM_T (4*nT)
#define NUM_V (N*B)


template <class real>
void meshdot_helper(int N, int B, int nT, int L,
                    real *T, real *dT, real *ph, real *dph, real *v, real *dv,
					int *inds, int *lens, int *shifts, int *perms, int is_perm, int pos, int mode, int threads)
{
    bool is_fd = (mode == 1), is_bd = (mode == 2); int tr = is_bd;

    // Split into a real and imaginary parts, which shows higher SIMD performance.
    real *T_r = RMALLOC(NUM_T*sizeof(real)), *T_i = RMALLOC(NUM_T*sizeof(real)),
         *v_r0 = RMALLOC(NUM_V*sizeof(real)), *v_i0 = RMALLOC(NUM_V*sizeof(real)), *temp_v0 = RMALLOC(NUM_V*sizeof(real)),
         *T_ph_r = RMALLOC(N*sizeof(real)), *T_ph_i = RMALLOC(N*sizeof(real)),
         *dT_r, *dT_i, *dv_r0, *dv_i0;

    split(T_r, T_i, T, NUM_T); split(v_r0, v_i0, v, NUM_V);

    if (is_fd || is_bd)
    {
        dT_r = RMALLOC(NUM_T*sizeof(real)); dT_i = RMALLOC(NUM_T*sizeof(real));
        dv_r0 = RMALLOC(NUM_V*sizeof(real)); dv_i0 = RMALLOC(NUM_V*sizeof(real));
        split(dT_r, dT_i, dT, NUM_T); split(dv_r0, dv_i0, dv, NUM_V);
    }
    real  *T00_r =  T_r,  *T01_r =  T_r + (1+tr)*nT,  *T10_r =  T_r + (2-tr)*nT,  *T11_r =  T_r + 3*nT,
          *T00_i =  T_i,  *T01_i =  T_i + (1+tr)*nT,  *T10_i =  T_i + (2-tr)*nT,  *T11_i =  T_i + 3*nT,
         *dT00_r = dT_r, *dT01_r = dT_r + (1)*nT,    *dT10_r = dT_r + (2)*nT,    *dT11_r = dT_r + 3*nT,
         *dT00_i = dT_i, *dT01_i = dT_i + (1)*nT,    *dT10_i = dT_i + (2)*nT,    *dT11_i = dT_i + 3*nT;

    if (is_bd)
    {
        pos = 1-pos;
        LOOP(k, NUM_T, T_i[k] = -T_i[k];) // T_i[:] = -T_i;
        LOOP(k, N,     ph[k]  = -ph[k]; ) // ph[:] = -ph;
    }
    int i = tr, j = 1-tr, n1, n2, dn;
    if (mode == 2) {n1 = L-1; n2 = -1; dn = -1;} else {n1 = 0; n2 = L; dn = 1;}
    // (n1, n2, dn) = (L-1, -1, -1) if (mode == 2) else (0, L, 1)

    const int MINBLOCK = 8;
    const int (BLOCKSIZE) = ((B - 1)/(threads * MINBLOCK) + 1)*MINBLOCK;

//    int l1 = 0, l2 = B;  // TODO -- for multithreaded code, may break into blocks.

    #pragma omp parallel for num_threads(threads)
    for (int l1 = 0; l1 < B; l1 += BLOCKSIZE)
    {
        int l2 = l1 + BLOCKSIZE; l2 = (l2 < B) ? l2 : B; int dl = l2 - l1;
        real *v_r = v_r0 + l1, *v_i = v_i0 + l1, *dv_r = dv_r0 + l1, *dv_i = dv_i0 + l1, *temp_v = temp_v0 + l1;

    // 3 potential steps: input screen, mesh, output screen.  Of course, there is only one phase screen.
    for (int step = 0; step < 3; step++)
    {
        if (step == 2*pos)
        {
            for (int k = 0; k < N; k++)
            {
                T_ph_r[k] = cos(ph[k]); T_ph_i[k] = sin(ph[k]);
                // T_ph_r = np.cos(ph).reshape((len(ph),1))
                // T_ph_i = np.sin(ph).reshape((len(ph),1))
            }

            for (int k = 0; k < N; k++)
            {
                //#pragma clang loop vectorize(assume_safety)
                for (int l = 0; l < dl; l++)
                {
                    int id_v = k*B + l, id_ph = k;
                    real temp = v_r[id_v] * T_ph_r[id_ph] - v_i[id_v] * T_ph_i[id_ph];
                    v_i[id_v] = v_r[id_v] * T_ph_i[id_ph] + v_i[id_v] * T_ph_r[id_ph];
                    v_r[id_v] = temp;
                    // temp   = v_r * T_ph_r - v_i * T_ph_i
                    // v_i[:] = v_r * T_ph_i + v_i * T_ph_r
                    // v_r[:] = temp

                    if (is_fd)
                    {
                        temp       = dv_r[id_v]*T_ph_r[id_ph] - dv_i[id_v]*T_ph_i[id_ph] - v_i[id_v]*dph[id_ph];
                        dv_i[id_v] = dv_r[id_v]*T_ph_i[id_ph] + dv_i[id_v]*T_ph_r[id_ph] + v_r[id_v]*dph[id_ph];
                        dv_r[id_v] = temp;
                        // temp    = dv_r*T_ph_r - dv_i*T_ph_i - v_i*dph.reshape((len(ph),1))
                        // dv_i[:] = dv_r*T_ph_i + dv_i*T_ph_r + v_r*dph.reshape((len(ph),1))
                        // dv_r[:] = temp
                    }
                    if (is_bd)
                    {
                        temp       = dv_r[id_v]*T_ph_r[id_ph] - dv_i[id_v]*T_ph_i[id_ph];
                        dv_i[id_v] = dv_r[id_v]*T_ph_i[id_ph] + dv_i[id_v]*T_ph_r[id_ph];
                        dv_r[id_v] = temp;
                        dph[id_ph] += v_r[id_v]*dv_i[id_v] - v_i[id_v]*dv_r[id_v];  // TODO make atomic op (multi-threaded)
                        // temp    = dv_r*T_ph_r - dv_i*T_ph_i
                        // dv_i[:] = dv_r*T_ph_i + dv_i*T_ph_r
                        // dv_r[:] = temp
                        // dph[:]  = np.real(v_r*dv_i - v_i*dv_r).sum(-1)
                    }
                }
            }
        }
        if (step == 1)
        {
            for (int n = n1; n != n2; n += dn)
            {
                int i1 = inds[n], l_n = lens[n], s = shifts[n], i2 = i1+l_n;

                if (is_perm)
                    permute(N, B, dl, is_fd || is_bd, perms+(n+int(is_bd))*N,
                            v_r, v_i, dv_r, dv_i, temp_v);
                    // p = perms[n + int(is_bd)]; v_r[:] = v_r[p]; v_i[:] = v_i[p]
                    // if is_fd or is_bd: dv_r[:] = dv_r[p]; dv_i[:] = dv_i[p]

                for (int k = 0; k < l_n; k++)
                {
                    int idx_T = i1+k;
                    // Odd how this vectorize command actually *slows down** the computation!

                    int iv1 = (s+2*k)*B;
                    real  *v1_r =  v_r + iv1,  *v1_i =  v_i + iv1,  *v2_r =  v1_r + B,  *v2_i =  v1_i + B,
                         *dv1_r = dv_r + iv1, *dv1_i = dv_i + iv1, *dv2_r = dv1_r + B, *dv2_i = dv1_i + B;
                    real T00k_r = T00_r[idx_T], T01k_r = T01_r[idx_T], T10k_r = T10_r[idx_T], T11k_r = T11_r[idx_T],
                         T00k_i = T00_i[idx_T], T01k_i = T01_i[idx_T], T10k_i = T10_i[idx_T], T11k_i = T11_i[idx_T],
                         dT00k_r = 0, dT01k_r = 0, dT10k_r = 0, dT11k_r = 0,
                         dT00k_i = 0, dT01k_i = 0, dT10k_i = 0, dT11k_i = 0;
                    if (is_fd)
                    {
                        dT00k_r = dT00_r[idx_T]; dT01k_r = dT01_r[idx_T];
                        dT10k_r = dT10_r[idx_T]; dT11k_r = dT11_r[idx_T];
                        dT00k_i = dT00_i[idx_T]; dT01k_i = dT01_i[idx_T];
                        dT10k_i = dT10_i[idx_T]; dT11k_i = dT11_i[idx_T];
                    }
                    // v1_r = v_r[s:s+2*l:2]; v2_r = v_r[s+1:s+2*l:2]
                    // v1_i = v_i[s:s+2*l:2]; v2_i = v_i[s+1:s+2*l:2]
                    // dv1_r = dv_r[s:s+2*l:2]; dv2_r = dv_r[s+1:s+2*l:2]
                    // dv1_i = dv_i[s:s+2*l:2]; dv2_i = dv_i[s+1:s+2*l:2]
                    // T00_r = T_r[0,0,i1:i2].reshape((l,1)); T01_r = T_r[i,j,i1:i2].reshape((l,1))
                    // T00_i = T_i[0,0,i1:i2].reshape((l,1)); T01_i = T_i[i,j,i1:i2].reshape((l,1))
                    // T10_r = T_r[j,i,i1:i2].reshape((l,1)); T11_r = T_r[1,1,i1:i2].reshape((l,1))
                    // T10_i = T_i[j,i,i1:i2].reshape((l,1)); T11_i = T_i[1,1,i1:i2].reshape((l,1))
                    // dT00_r = dT_r[0,0,i1:i2].reshape((l,1)); dT01_r = dT_r[i,j,i1:i2].reshape((l,1))
                    // dT00_i = dT_i[0,0,i1:i2].reshape((l,1)); dT01_i = dT_i[i,j,i1:i2].reshape((l,1))
                    // dT10_r = dT_r[j,i,i1:i2].reshape((l,1)); dT11_r = dT_r[1,1,i1:i2].reshape((l,1))
                    // dT10_i = dT_i[j,i,i1:i2].reshape((l,1)); dT11_i = dT_i[1,1,i1:i2].reshape((l,1))


                    if (is_fd)
                    {
                        #pragma clang loop vectorize(assume_safety)
                        for (int l = 0; l < dl; l++)
                        {
                            real temp_1r, temp_1i, temp_2r;

                            // dv -> dT*v + T*dv
                            temp_1r  = dT00k_r*v1_r[l] + dT01k_r*v2_r[l] + T00k_r*dv1_r[l] + T01k_r*dv2_r[l]
                                     - dT00k_i*v1_i[l] - dT01k_i*v2_i[l] - T00k_i*dv1_i[l] - T01k_i*dv2_i[l];
                            temp_1i  = dT00k_r*v1_i[l] + dT01k_r*v2_i[l] + T00k_r*dv1_i[l] + T01k_r*dv2_i[l]
                                     + dT00k_i*v1_r[l] + dT01k_i*v2_r[l] + T00k_i*dv1_r[l] + T01k_i*dv2_r[l];
                            temp_2r  = dT10k_r*v1_r[l] + dT11k_r*v2_r[l] + T10k_r*dv1_r[l] + T11k_r*dv2_r[l]
                                     - dT10k_i*v1_i[l] - dT11k_i*v2_i[l] - T10k_i*dv1_i[l] - T11k_i*dv2_i[l];
                            dv2_i[l] = dT10k_r*v1_i[l] + dT11k_r*v2_i[l] + T10k_r*dv1_i[l] + T11k_r*dv2_i[l]
                                     + dT10k_i*v1_r[l] + dT11k_i*v2_r[l] + T10k_i*dv1_r[l] + T11k_i*dv2_r[l];
                            dv2_r[l] = temp_2r;
                            dv1_r[l] = temp_1r;
                            dv1_i[l] = temp_1i;
                            // temp_1r  = dT00_r*v1_r+dT01_r*v2_r+T00_r*dv1_r+T01_r*dv2_r-dT00_i*v1_i-dT01_i*v2_i-T00_i*dv1_i-T01_i*dv2_i
                            // temp_1i  = dT00_r*v1_i+dT01_r*v2_i+T00_r*dv1_i+T01_r*dv2_i+dT00_i*v1_r+dT01_i*v2_r+T00_i*dv1_r+T01_i*dv2_r
                            // temp_2r  = dT10_r*v1_r+dT11_r*v2_r+T10_r*dv1_r+T11_r*dv2_r-dT10_i*v1_i-dT11_i*v2_i-T10_i*dv1_i-T11_i*dv2_i
                            // dv2_i[:] = dT10_r*v1_i+dT11_r*v2_i+T10_r*dv1_i+T11_r*dv2_i+dT10_i*v1_r+dT11_i*v2_r+T10_i*dv1_r+T11_i*dv2_r
                            // dv2_r[:] = temp_2r
                            // dv1_r[:] = temp_1r
                            // dv1_i[:] = temp_1i
                        }
                    }

                    #pragma clang loop vectorize(assume_safety)
                    for (int l = 0; l < dl; l++)
                    {
                        real temp_1r, temp_1i, temp_2r;

                        // # v -> T*v
                        temp_1r  = T00k_r*v1_r[l] + T01k_r*v2_r[l] - T00k_i*v1_i[l] - T01k_i*v2_i[l];
                        temp_1i  = T00k_r*v1_i[l] + T01k_r*v2_i[l] + T00k_i*v1_r[l] + T01k_i*v2_r[l];
                        temp_2r  = T10k_r*v1_r[l] + T11k_r*v2_r[l] - T10k_i*v1_i[l] - T11k_i*v2_i[l];
                        v2_i[l]  = T10k_r*v1_i[l] + T11k_r*v2_i[l] + T10k_i*v1_r[l] + T11k_i*v2_r[l];
                        v2_r[l]  = temp_2r;
                        v1_r[l]  = temp_1r;
                        v1_i[l]  = temp_1i;
                        // temp_1r  = T00_r*v1_r + T01_r*v2_r - T00_i*v1_i - T01_i*v2_i
                        // temp_1i  = T00_r*v1_i + T01_r*v2_i + T00_i*v1_r + T01_i*v2_r
                        // temp_2r  = T10_r*v1_r + T11_r*v2_r - T10_i*v1_i - T11_i*v2_i
                        // v2_i[:]   = T10_r*v1_i + T11_r*v2_i + T10_i*v1_r + T11_i*v2_r
                        // v2_r[:]   = temp_2r
                        // v1_r[:]   = temp_1r
                        // v1_i[:]   = temp_1i
                    }

                    if (is_bd)
                    {
                        #pragma clang loop vectorize(assume_safety)
                        for (int l = 0; l < dl; l++)
                        {
                            real temp_1r, temp_1i, temp_2r;

                            // dJ/dT_ij = (dJ/dy_i)* x_j
                            dT00k_r += (dv1_r[l]*v1_r[l] + dv1_i[l]*v1_i[l]);
                            dT00k_i += (dv1_r[l]*v1_i[l] - dv1_i[l]*v1_r[l]);
                            dT01k_r += (dv1_r[l]*v2_r[l] + dv1_i[l]*v2_i[l]);
                            dT01k_i += (dv1_r[l]*v2_i[l] - dv1_i[l]*v2_r[l]);
                            dT10k_r += (dv2_r[l]*v1_r[l] + dv2_i[l]*v1_i[l]);
                            dT10k_i += (dv2_r[l]*v1_i[l] - dv2_i[l]*v1_r[l]);
                            dT11k_r += (dv2_r[l]*v2_r[l] + dv2_i[l]*v2_i[l]);
                            dT11k_i += (dv2_r[l]*v2_i[l] - dv2_i[l]*v2_r[l]);

                            // dT_r[0,0,i1:i2] = (dv1_r*v1_r + dv1_i*v1_i).sum(-1); dT_i[0,0,i1:i2] = (dv1_r*v1_i - dv1_i*v1_r).sum(-1)
                            // dT_r[0,1,i1:i2] = (dv1_r*v2_r + dv1_i*v2_i).sum(-1); dT_i[0,1,i1:i2] = (dv1_r*v2_i - dv1_i*v2_r).sum(-1)
                            // dT_r[1,0,i1:i2] = (dv2_r*v1_r + dv2_i*v1_i).sum(-1); dT_i[1,0,i1:i2] = (dv2_r*v1_i - dv2_i*v1_r).sum(-1)
                            // dT_r[1,1,i1:i2] = (dv2_r*v2_r + dv2_i*v2_i).sum(-1); dT_i[1,1,i1:i2] = (dv2_r*v2_i - dv2_i*v2_r).sum(-1)

                            // dv -> T*dv
                            temp_1r  = T00k_r*dv1_r[l] + T01k_r*dv2_r[l] - T00k_i*dv1_i[l] - T01k_i*dv2_i[l];
                            temp_1i  = T00k_r*dv1_i[l] + T01k_r*dv2_i[l] + T00k_i*dv1_r[l] + T01k_i*dv2_r[l];
                            temp_2r  = T10k_r*dv1_r[l] + T11k_r*dv2_r[l] - T10k_i*dv1_i[l] - T11k_i*dv2_i[l];
                            dv2_i[l] = T10k_r*dv1_i[l] + T11k_r*dv2_i[l] + T10k_i*dv1_r[l] + T11k_i*dv2_r[l];
                            dv2_r[l] = temp_2r;
                            dv1_r[l] = temp_1r;
                            dv1_i[l] = temp_1i;
                            // temp_1r  = T00_r*dv1_r + T01_r*dv2_r - T00_i*dv1_i - T01_i*dv2_i
                            // temp_1i  = T00_r*dv1_i + T01_r*dv2_i + T00_i*dv1_r + T01_i*dv2_r
                            // temp_2r  = T10_r*dv1_r + T11_r*dv2_r - T10_i*dv1_i - T11_i*dv2_i
                            // dv2_i[:] = T10_r*dv1_i + T11_r*dv2_i + T10_i*dv1_r + T11_i*dv2_r
                            // dv2_r[:] = temp_2r
                            // dv1_r[:] = temp_1r
                            // dv1_i[:] = temp_1i
                        }
                    }
                    if (is_bd)
                    {
                        dT00_r[idx_T] += dT00k_r; dT00_i[idx_T] += dT00k_i;
                        dT01_r[idx_T] += dT01k_r; dT01_i[idx_T] += dT01k_i;
                        dT10_r[idx_T] += dT10k_r; dT10_i[idx_T] += dT10k_i;
                        dT11_r[idx_T] += dT11k_r; dT11_i[idx_T] += dT11k_i;
                    }
                }
            }

            if (is_perm)
                permute(N, B, dl, is_fd || is_bd, perms+(is_bd ? 0 : L)*N,
                        v_r, v_i, dv_r, dv_i, temp_v);
                // p = perms[0 if is_bd else L]; v_r[:] = v_r[p]; v_i[:] = v_i[p]
                // if is_fd or is_bd: dv_r[:] = dv_r[p]; dv_i[:] = dv_i[p]
        }
    }

    }

    // Combine to the packed complex format used in NumPy.
    merge(T, T_r, T_i, NUM_T); merge(v, v_r0, v_i0, NUM_V);
    free(T_r); free(T_i); free(v_r0); free(v_i0); free(temp_v0); free(T_ph_r); free(T_ph_i);
    if (is_fd || is_bd)
    {
        merge(dT, dT_r, dT_i, NUM_T); merge(dv, dv_r0, dv_i0, NUM_V);
        free(dT_r); free(dT_i); free(dv_r0); free(dv_i0);
    }
    if (is_bd) {LOOP(k, N, ph[k] = -ph[k]; )} // ph[:] = -ph;
}

extern "C"
{
    void meshdot_helper32(int N, int B, int nT, int L,
                          float *T, float *dT, float *ph, float *dph, float *v, float *dv,
			              int *inds, int *lens, int *shifts, int *perms, int is_perm, int pos, int mode, int threads)
    {
        meshdot_helper<float>(N, B, nT, L, T, dT, ph, dph, v, dv, inds, lens, shifts, perms, is_perm, pos, mode, threads);
    }

    void meshdot_helper64(int N, int B, int nT, int L,
                          double *T, double *dT, double *ph, double *dph, double *v, double *dv,
			              int *inds, int *lens, int *shifts, int *perms, int is_perm, int pos, int mode, int threads)
    {
        meshdot_helper<double>(N, B, nT, L, T, dT, ph, dph, v, dv, inds, lens, shifts, perms, is_perm, pos, mode,
                               threads);
    }
}