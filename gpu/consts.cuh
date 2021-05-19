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
    #define L_ker (L0)  // Actual number of layers stored in the kernel = L0
    #define L_preload (L0*nL)  // Number of shifts / lens pre-loaded.
    #if   CROSSING_TYPE == MZI
        #define stride_T       4
        #define stride_dT      4
        #define define_T       __shared__ complex64 T[L0][4*K][32]
        #define define_T_dT    __shared__ complex64 T[L0][4*K][32], dT[L0][4*K][32]
        #define load_u         load_u_mzi(u, u_in)
        #define load_u_du      load_u_du_mzi(u, du, u_in, du_in)
        #define load_u_du_bk   load_u_du_mzi(u, dJdu, u_out, dJdu_out)
        #define load_T         load_T_mzi
        #define load_T_dT      load_T_dT_mzi
        #define load_T_dT_bk   load_T_dT_bk_mzi
        #define save_u         save_u_mzi(u, u_out)
        #define save_u_du      save_u_du_mzi(u, du, u_out, du_out)
        #define save_u_du_bk   save_u_du_mzi(u, dJdu, u_in,  dJdu_in)
        #define save_dp        save_dp_mzi
        #define matmult        matmult_mzi
        #define matmult_d      matmult_d_mzi
        #define matmult_bk     matmult_bk_mzi
        #define scalar         complex64
    #elif CROSSING_TYPE == SYM
        #define stride_T       3
        #define stride_dT      3
        #define define_T       __shared__ float T[L0][3*K][32]
        #define define_T_dT    __shared__ float T[L0][3*K][32], dT[L0][3*K][32]
        #define load_u         load_u_sym(u, u_in)
        #define load_u_du      load_u_du_sym(u, du, u_in, du_in)
        #define load_u_du_bk   load_u_du_sym(u, dJdu, u_out, dJdu_out)
        #define load_T         load_T_sym
        #define load_T_dT      load_T_dT_sym
        #define load_T_dT_bk   load_T_dT_bk_sym
        #define save_u         save_u_sym(u, u_out)
        #define save_u_du      save_u_du_sym(u, du, u_out, du_out)
        #define save_u_du_bk   save_u_du_sym(u, dJdu, u_in,  dJdu_in)
        #define save_dp        save_dp_sym
        #define matmult        matmult_sym
        #define matmult_d      matmult_d_sym
        #define matmult_bk     matmult_bk_sym
        #define scalar         complex64
    #else
        #define stride_T       2
        #define stride_dT      1
        #define dth            dT
        #define stride_dth     stride_dT
        #define define_T       __shared__ float T[L0][2*K][32]
        #define define_T_dT    __shared__ float T[L0][2*K][32], dth[L0][K][32]
        #define load_u         load_u_orth(u, u_in)
        #define load_u_du      load_u_du_orth(u, du, u_in, du_in)
        #define load_u_du_bk   load_u_du_orth(u, dJdu, u_out, dJdu_out)
        #define load_T         load_T_orth
        #define load_T_dT      load_T_dT_orth
        #define load_T_dT_bk   load_T_dT_bk_orth
        #define save_u         save_u_orth(u, u_out)
        #define save_u_du      save_u_du_orth(u, du, u_out, du_out)
        #define save_u_du_bk   save_u_du_orth(u, dJdu, u_in,  dJdu_in)
        #define save_dp        save_dp_orth
        #define matmult        matmult_orth
        #define matmult_d      matmult_d_orth
        #define matmult_bk     matmult_bk_orth
        #define scalar         float
    #endif
#else
    #undef CONSTANTS_DEFINED
    #undef L_ker
    #undef L_preload
    #undef K
    #undef L0
    #undef nL
    #undef fname
    #undef stride_T
    #undef stride_dT
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
        #undef stride_dth
    #endif
#endif