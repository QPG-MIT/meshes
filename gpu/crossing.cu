// meshes/gpu/backdiff.cu
// Ryan Hamerly, 4/6/21
//
// Device functions to handle the MZI arithmetic, namely obtaining (T, dT) from (p, dp, s) and back-propagating
// gradients of dT to the parameters p.
//
// History:
//   04/06/21: Moved this to its own file, added matmult_** routines and dp_mzi (for backprop).

// Initializes an identity transfer matrix [[1, 0], [0, 1]].
__device__ void Tij_identity(complex64 *T, complex64 *dT)
{
    T[0] = 1; T[32] = 0; T[64] = 0; T[96] = 1;
    if (dT)
    {
        dT[0] = 0; dT[32] = 0; dT[64] = 0; dT[96] = 0;
    }
}

// Initializes T = [T11, T12, T21, T22] to given MZI settings (θ, φ) and imperfections (α, β).
__device__ void Tij_mzi(const float *p, const float *dp, const float *s, complex64 *T, complex64 *dT, 
                        float ps_cache[4], bool init_dT)
{
    if (ps_cache)
    {
        ps_cache[0] = p[0]; ps_cache[1] = p[1]; 
        ps_cache[2] = s[0]; ps_cache[3] = s[1];
    }  

	// cos(θ/2), sin(θ/2), cos(θ/2+φ), sin(θ/2+φ)
	float C, S, C1, S1;
	__sincosf(p[0]/2,      &S , &C );
	__sincosf(p[0]/2+p[1], &S1, &C1);

	// cos(α ± β), sin(α ± β)
	float Cp, Sp, Cm, Sm;
	__sincosf(s[0]+s[1],   &Sp, &Cp);
	__sincosf(s[0]-s[1],   &Sm, &Cm);

	// Equivalent Python code:
    // (Cp, Cm, C, Sp, Sm, S) = [fn(x) for fn in [np.cos, np.sin] for x in [beta[0]+beta[1], beta[0]-beta[1], theta/2]]
    // T = np.exp(1j*theta/2) * np.array([[np.exp(1j*phi) * (1j*S*Cm - C*Sp),    1j*C*Cp - S*Sm],
    //                                    [np.exp(1j*phi) * (1j*C*Cp + S*Sm),   -1j*S*Cm - C*Sp]])
    T[0 ] = complex64(C1, S1) * complex64(-C*Sp,  S*Cm);
    T[32] = complex64(C , S ) * complex64(-S*Sm,  C*Cp);
    T[64] = complex64(C1, S1) * complex64( S*Sm,  C*Cp);
    T[96] = complex64(C , S ) * complex64(-C*Sp, -S*Cm);

    // Equivalent Python code:
    // dT = (np.exp(1j*np.array([[[phi+theta, theta]]])) *
    //       np.array([[[1j*(1j*S*Cm-C*Sp)+( 1j*C*Cm+S*Sp),   1j*( 1j*C*Cp-S*Sm)+(-1j*S*Cp-C*Sm)],
    //                  [1j*(1j*C*Cp+S*Sm)+(-1j*S*Cp+C*Sm),   1j*(-1j*S*Cm-C*Sp)+(-1j*C*Cm+S*Sp)]],
    //                 [[1j*(1j*S*Cm-C*Sp),                   0*S                               ],
    //                  [1j*(1j*C*Cp+S*Sm),                   0*S                               ]]]))
    if (dT)
    {
        if (init_dT)
        {
            dT[0 ] = complex64(C1, S1) * (0.5f * dp[0] * (Cm-Sp) * complex64(-S,  C) + dp[1] * complex64(-Cm*S, -C*Sp));
            dT[32] = complex64(C , S ) * (0.5f * dp[0] * (Cp+Sm) * complex64(-C, -S));
            dT[64] = complex64(C1, S1) * (0.5f * dp[0] * (Cp-Sm) * complex64(-C, -S) + dp[1] * complex64(-C*Cp,  S*Sm));
            dT[96] = complex64(C , S ) * (0.5f * dp[0] * (Cm+Sp) * complex64( S, -C));
        }
        else
        {
            dT[0] = 0; dT[32] = 0; dT[64] = 0; dT[96] = 0;
        }
    }
}

// Gets the gradients with respect to MZI parameters p = (θ, φ).  Only used in back-propagation.
// dJ/dp = dJ/dT_{ij} dT_{ij}/dp + c.c. = 2*Re[dJ/dT_{ij} dT_{ij}/dp]
__device__ void dp_mzi(const float *p, const float *s, const complex64 dT[4], float *dp)
{
	// cos(θ/2), sin(θ/2), cos(θ/2+φ), sin(θ/2+φ)
	float C, S, C1, S1;
	__sincosf(p[0]/2,      &S , &C );
	__sincosf(p[0]/2+p[1], &S1, &C1);

	// cos(α ± β), sin(α ± β)
	float Cp, Sp, Cm, Sm;
	__sincosf(s[0]+s[1],   &Sp, &Cp);
	__sincosf(s[0]-s[1],   &Sm, &Cm);

    // TODO -- simplify this once I'm sure that it works.
    float dp0, dp1;
    dp0 = (dT[0 ] * complex64(C1, S1) * (0.5f * (Cm-Sp) * complex64(-S,  C)) + 
           dT[32] * complex64(C , S ) * (0.5f * (Cp+Sm) * complex64(-C, -S)) + 
           dT[64] * complex64(C1, S1) * (0.5f * (Cp-Sm) * complex64(-C, -S)) + 
           dT[96] * complex64(C , S ) * (0.5f * (Cm+Sp) * complex64( S, -C))).real() * 2;
    dp1 = (dT[0 ] * complex64(C1, S1) * (complex64(-Cm*S, -C*Sp)) + 
           dT[64] * complex64(C1, S1) * (complex64(-C*Cp,  S*Sm))).real() * 2;
    atomicAdd(&dp[0], dp0);
    atomicAdd(&dp[1], dp1);
}

__device__ __inline__ void matmult(const complex64 T[4], complex64 &u1, complex64 &u2, complex64 &temp, bool cond)
{
    // u_i -> T_{ij} u_j
    temp = T[0 ]*u1 + T[32]*u2;
    u2   = T[64]*u1 + T[96]*u2;
    if (cond)
        u1 = temp;
}

__device__ __inline__ void matmult_d(const complex64 *T, const complex64 *dT, 
                                     complex64 &u1, complex64 &u2, complex64 &du1, complex64 &du2, 
                                     complex64 &temp, bool cond)
{
    // du_i -> T_{ij} du_j + dT_{ij} u_j
    temp = T[0 ]*du1 + T[32]*du2 + dT[0 ]*u1 + dT[32]*u2;
    du2  = T[64]*du1 + T[96]*du2 + dT[64]*u1 + dT[96]*u2;
    if (cond)
        du1 = temp;
    // u_i -> T_{ij} u_j
    temp = T[0 ]*u1 + T[32]*u2;
    u2   = T[64]*u1 + T[96]*u2;
    if (cond)
        u1 = temp;
}

__device__ __inline__ void atomicAdd(complex64 *A, complex64 B)
{
    float *A_float = (float *) A;
    atomicAdd(A_float,   B.real());
    atomicAdd(A_float+1, B.imag());
}

// Back-propagation of signals and gradients.
// Here, (dJdu1, dJdu2) represent the gradients dJ/du*, which is conjugate to dJ/du.
// TODO: check that conj(A)*B is properly compiled (takes as many FLOPS as A*B).  Otherwise, pre-conjugate T.
__device__ __inline__ void matmult_bk(const complex64 *T, complex64 *dT, 
                                      complex64 &u1, complex64 &u2, complex64 &dJdu1, complex64 &dJdu2, 
                                      complex64 &temp, bool cond)
{
    // u_i -> (T^dag)_{ij} u_j = (T_{ji})^* u_j
    temp = conj(T[0 ])*u1 + conj(T[64])*u2;
    u2   = conj(T[32])*u1 + conj(T[96])*u2;
    if (cond)
        u1 = temp;
    // dJ/dT_{ij} = (dJ/du_i)_{out} (u_j)_{in} = (dJ/du_i^*)_{out}^* (u_j)_{in} 
    atomicAdd(&dT[0 ], conj(dJdu1)*u1); atomicAdd(&dT[32], conj(dJdu1)*u2); 
    atomicAdd(&dT[64], conj(dJdu2)*u1); atomicAdd(&dT[96], conj(dJdu2)*u2); 
    //dT[0 ] += conj(dJdu1)*u1;  dT[32] += conj(dJdu1)*u2;
    //dT[64] += conj(dJdu2)*u1;  dT[96] += conj(dJdu2)*u2;
    // dJ/du_i^* -> (T^*)_{ij} dJ/du_j = (T_{ji}) dJ/du_j^*
    temp  = conj(T[0 ])*dJdu1 + conj(T[64])*dJdu2;
    dJdu2 = conj(T[32])*dJdu1 + conj(T[96])*dJdu2;
    if (cond)
        dJdu1 = temp;
}
