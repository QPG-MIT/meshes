// meshes/gpu/consts.cuh
// Ryan Hamerly, 5/19/21
//
// Sets constants and macros for the CUDA kernel as a function of crossing type (MZI | SYM | ORTH) and
// mesh type (regular | FFT).
//
// History:
//   05/19/21: Created this file, ported macros from fwdprop.cu, fwddiff.cu, backdiff.cu.


#ifndef CONSTANTS_DEFINED
    #define CONSTANTS_DEFINED 1
    
    #define L_ker (L0)          // Actual number of layers stored in the kernel = L0
    #define L_preload (L0*nL)   // Number of shifts / lens pre-loaded.

    #if   K == 1
        #define P 1
    #elif K == 2
        #define P 2
    #elif K == 4
        #define P 3
    #elif K == 8
        #define P 4
    #elif K == 16
        #define P 5
    #else
        #define P kaboom
    #endif

    #if FFT
        #define geom_pars           int *strides
        #define define_cache        __shared__ int strides_cache[nL*L0]
        #define load_cache_fwd      load_strides_cache_fwd
        #define load_cache_rev      load_strides_cache_rev
    #else
        #define geom_pars           int *lens, int *shifts
        #define define_cache        __shared__ int shifts_cache[nL*L0], lens_cache[nL*L0]
        #define load_cache_fwd      load_pos_cache_fwd
        #define load_cache_rev      load_pos_cache_rev
    #endif

    #if   CROSSING_TYPE == MZI
        #define s_T                 4
        #define s_dT                4
        #define define_T            __shared__ complex64 T[L0][4*K][32]
        #define define_T_dT         __shared__ complex64 T[L0][4*K][32], dT[L0][4*K][32]
        #define load_u              load_u_mzi(u, u_in)
        #define load_u_du           load_u_du_mzi(u, du, u_in, du_in)
        #define load_u_du_bk        load_u_du_mzi(u, dJdu, u_out, dJdu_out)
        #if FFT
            #define load_T          loadft_T_mzi
            #define load_T_dT       loadft_T_dT_mzi
            #define load_T_dT_bk    loadft_T_dT_bk_mzi
            #define save_dp         saveft_dp_mzi
        #else
            #define load_T          load_T_mzi
            #define load_T_dT       load_T_dT_mzi
            #define load_T_dT_bk    load_T_dT_bk_mzi
            #define save_dp         save_dp_mzi
        #endif
        #define save_u              save_u_mzi(u, u_out)
        #define save_u_du           save_u_du_mzi(u, du, u_out, du_out)
        #define save_u_du_bk        save_u_du_mzi(u, dJdu, u_in,  dJdu_in)
        #define matmult             matmult_mzi
        #define matmult_d           matmult_d_mzi
        #define matmult_bk          matmult_bk_mzi
        #define scalar              complex64
    #elif CROSSING_TYPE == SYM
        #define s_T                 3
        #define s_dT                3
        #define define_T            __shared__ float T[L0][3*K][32]
        #define define_T_dT         __shared__ float T[L0][3*K][32], dT[L0][3*K][32]
        #define load_u              load_u_sym(u, u_in)
        #define load_u_du           load_u_du_sym(u, du, u_in, du_in)
        #define load_u_du_bk        load_u_du_sym(u, dJdu, u_out, dJdu_out)
        #if FFT
            #define load_T          loadft_T_sym
            #define load_T_dT       loadft_T_dT_sym
            #define load_T_dT_bk    loadft_T_dT_bk_sym
            #define save_dp         saveft_dp_sym
        #else
            #define load_T          load_T_sym
            #define load_T_dT       load_T_dT_sym
            #define load_T_dT_bk    load_T_dT_bk_sym
            #define save_dp         save_dp_sym
        #endif
        #define save_u              save_u_sym(u, u_out)
        #define save_u_du           save_u_du_sym(u, du, u_out, du_out)
        #define save_u_du_bk        save_u_du_sym(u, dJdu, u_in,  dJdu_in)
        #define matmult             matmult_sym
        #define matmult_d           matmult_d_sym
        #define matmult_bk          matmult_bk_sym
        #define scalar              complex64
    #elif CROSSING_TYPE == ORTH
        #define s_T                 2
        #define s_dT                1
        #define dth                 dT
        #define s_dth               s_dT
        #define define_T            __shared__ float T[L0][2*K][32]
        #define define_T_dT         __shared__ float T[L0][2*K][32], dth[L0][K][32]
        #define load_u              load_u_orth(u, u_in)
        #define load_u_du           load_u_du_orth(u, du, u_in, du_in)
        #define load_u_du_bk        load_u_du_orth(u, dJdu, u_out, dJdu_out)
        #if FFT
            #define load_T          loadft_T_orth
            #define load_T_dT       loadft_T_dT_orth
            #define load_T_dT_bk    loadft_T_dT_bk_orth
            #define save_dp         saveft_dp_orth
        #else
            #define load_T          load_T_orth
            #define load_T_dT       load_T_dT_orth
            #define load_T_dT_bk    load_T_dT_bk_orth
            #define save_dp         save_dp_orth
        #endif
        #define save_u              save_u_orth(u, u_out)
        #define save_u_du           save_u_du_orth(u, du, u_out, du_out)
        #define save_u_du_bk        save_u_du_orth(u, dJdu, u_in,  dJdu_in)
        #define matmult             matmult_orth
        #define matmult_d           matmult_d_orth
        #define matmult_bk          matmult_bk_orth
        #define scalar              float
    #endif
#else
    #undef CONSTANTS_DEFINED
    #undef L_ker
    #undef L_preload
    #undef P
    #undef K
    #undef L0
    #undef nL
    #undef fname
    #undef geom_pars
    #undef define_cache
    #undef load_cache_fwd
    #undef load_cache_rev
    #undef s_T
    #undef s_dT
    #undef define_T
    #undef define_T_dT
    #undef load_u
    #undef load_u_du
    #undef load_u_du_bk
    #undef load_T
    #undef load_T_dT
    #undef load_T_dT_bk
    #undef save_u
    #undef save_u_du
    #undef save_u_du_bk
    #undef save_dp
    #undef matmult
    #undef matmult_d
    #undef matmult_bk
    #undef scalar
    #if CROSSING_TYPE == ORTH
        #undef dth
        #undef s_dth
    #endif
#endif