// meshes/gpu/backdiff.cu
// Ryan Hamerly, 4/6/21
//
// Device functions to handle the MZI arithmetic, namely obtaining (T, dT) from (p, dp, s) and back-propagating
// gradients of dT to the parameters p.
//
// History:
//   04/06/21: Moved this to its own file, added matmult_** routines and dp_mzi (for backprop).
//   04/10/21: Added symmetric and orthogonal representations.

// Initializes an identity transfer matrix [[1, 0], [0, 1]].
__device__ void Tij_identity(complex64 *T, complex64 *dT)
{
    T[0] = 1; T[32] = 0; T[64] = 0; T[96] = 1;
    if (dT)
    {
        dT[0] = 0; dT[32] = 0; dT[64] = 0; dT[96] = 0;
    }
}
__device__ void Tij_identity_sym(float *T, float *dT)
{
    T[0] = 1; T[32] = 0; T[64] = 0;
    if (dT)
    {
        dT[0] = 0; dT[32] = 0; dT[64] = 0;
    }
}
__device__ void Tij_identity_orth(float *T, float *dth)
{
    T[0] = 1; T[32] = 0;
    if (dth)
        dth[0] = 0;
}

// Initializes T = [T11, T12, T21, T22] to given MZI settings (θ, φ) and imperfections (α, β).
__device__ void Tij_mzi(const float *p, const float *dp, const float *s, complex64 *T, complex64 *dT, bool init_dT)
{
	// cos(θ/2), sin(θ/2), cos(θ/2+φ), sin(θ/2+φ)
	float C, S, C1, S1;
	__sincosf(0.5f*p[0],   &S , &C );
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

// Symmetric MZI design.
//
// T = [[exp(+i*φ) (sin(θ/2) + i cos(θ/2)sin(2α)),  i cos(θ/2)cos(2α)                       ],
//      [i cos(θ/2)cos(2α),                         exp(-i*φ) (sin(θ/2) - i cos(θ/2)sin(2α))]]
__device__ void Tij_mzi_sym(const float *p, const float *dp, const float *s, float *T, float *dT, int cartesian, bool init_dT)
{
    if (cartesian == 0)
    {
        // cos(θ/2), sin(θ/2), cos(φ), sin(φ)
        float C_th, S_th, C_ph, S_ph;
        __sincosf(0.5f*p[0], &S_th, &C_th);
        __sincosf(p[1],      &S_ph, &C_ph);

        // cos(2α), sin(2α), the imperfections (if any)
        float C_2a = 1, S_2a = 0;
        if (s != 0)
            __sincosf(2*s[0], &S_2a, &C_2a);

        T[0 ] = S_th*C_ph - C_th*S_ph*S_2a;
        T[32] = S_th*S_ph + C_th*C_ph*S_2a;
        T[64] = C_th*C_2a;

        if (dT)
        {
            if (init_dT)
            {
                dT[0 ] =  0.5f*dp[0]*(C_th*C_ph + S_th*S_ph*S_2a) - dp[1]*(S_th*S_ph + C_th*C_ph*S_2a);
                dT[32] =  0.5f*dp[0]*(C_th*S_ph - S_th*C_ph*S_2a) + dp[1]*(S_th*C_ph - C_th*S_ph*S_2a);
                dT[64] = -0.5f*dp[0]*S_th*C_2a;
            }
            else
            {
                dT[0] = 0; dT[32] = 0; dT[64] = 0;
            }
        }
    }
    else
    {
        // TODO -- handle case where cartesian=true        
    }
}

// Orthogonal MZI design.  Builds up SO(N), not U(N).
//
// T = [[sin(θ/2), -cos(θ/2)],
//      [cos(θ/2),  sin(θ/2)]]
__device__ void Tij_mzi_orth(const float *p, const float *dp, float *T, float *dth, bool init_dT)
{
	// cos(θ/2), sin(θ/2)
	float C, S;
	__sincosf(p[0]/2, &S, &C);
    
    T[0 ] =  S;
    T[32] = -C;

    if (dth)
    {
        if (init_dT)
            dth[0] = 0.5f*dp[0];
        else
            dth[0] = 0.0f;
    }
}

// Gets the gradients with respect to MZI parameters p = (θ, φ).  Only used in back-propagation.
// dJ/dp = dJ/dT_{ij} dT_{ij}/dp + c.c. = 2*Re[dJ/dT_{ij} dT_{ij}/dp]
__device__ void dp_mzi(const float *p, const float *s, const complex64 *dT, float *dp)
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
           dT[96] * complex64(C , S ) * (0.5f * (Cm+Sp) * complex64( S, -C))).real();
    dp1 = (dT[0 ] * complex64(C1, S1) * (complex64(-Cm*S, -C*Sp)) + 
           dT[64] * complex64(C1, S1) * (complex64(-C*Cp,  S*Sm))).real();
    atomicAdd(&dp[0], dp0);
    atomicAdd(&dp[1], dp1);
}
__device__ void dp_mzi_sym(const float *p, const float *s, const float *dT, float *dp)
{
    // Initialize cos(...), sin(...).
	float C_th, S_th, C_ph, S_ph, C_2a = 1, S_2a = 0;
	__sincosf(p[0]/2, &S_th, &C_th);
	__sincosf(p[1],   &S_ph, &C_ph);
    if (s != 0)
        __sincosf(2*s[0], &S_2a, &C_2a);
    
    float dp0, dp1;
    dp0 = (dT[0 ] * (C_th*C_ph + S_th*S_ph*S_2a) + 
           -dT[32] * (C_th*S_ph - S_th*C_ph*S_2a) - 
           -dT[64] * S_th*C_2a) * 0.5f;
    dp1 = (-dT[0 ] * (S_th*S_ph + C_th*C_ph*S_2a) + 
            -dT[32] * (S_th*C_ph - C_th*S_ph*S_2a)) * 1.0f;
    atomicAdd(&dp[0], dp0);
    atomicAdd(&dp[1], dp1);
}
__device__ void dp_mzi_orth(const float *dth, float *dp)
{
    atomicAdd(&dp[0], 0.5f*dth[0]);
}

__device__ __inline__ void matmult(const complex64 *T, complex64 &u1, complex64 &u2, complex64 &temp, bool cond)
{
    // u_i -> T_{ij} u_j
    temp = T[0 ]*u1 + T[32]*u2;
    u2   = T[64]*u1 + T[96]*u2;
    if (cond)
        u1 = temp;
}
__device__ __inline__ void matmult_sym(const float *T, float &u1, float &v1, float &u2, float &v2, 
                                       float &temp1, float &temp2, float &temp3, bool cond)
{
    // u_i -> T_{ij} u_j
    temp1 =  T[0 ]*u1 - T[32]*v1 - T[64]*v2;
    temp2 =  T[32]*u1 + T[0 ]*v1 + T[64]*u2;
    temp3 =  T[0 ]*u2 + T[32]*v2 - T[64]*v1;
    v2    = -T[32]*u2 + T[0 ]*v2 + T[64]*u1;
    u2    =  temp3;
    if (cond)
    {
        u1 = temp1;
        v1 = temp2;
    }
}
__device__ __inline__ void matmult_orth(const float *T, float &u1, float &u2, float &temp, bool cond)
{
    temp =  T[0 ]*u1 + T[32]*u2;
    u2   = -T[32]*u1 + T[0 ]*u2;
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
    matmult(T, u1, u2, temp, cond);
}
__device__ __inline__ void matmult_d_sym(const float *T, const float *dT, float &u1, float &v1, float &u2, float &v2, 
                                         float &du1, float &dv1, float &du2, float &dv2,
                                         float &temp1, float &temp2, float &temp3, bool cond)
{
    // du_i -> T_{ij} du_j + dT_{ij} u_j
    temp1 =  T[0 ]*du1 - T[32]*dv1 - T[64]*dv2 + dT[0 ]*u1 - dT[32]*v1 - dT[64]*v2;
    temp2 =  T[32]*du1 + T[0 ]*dv1 + T[64]*du2 + dT[32]*u1 + dT[0 ]*v1 + dT[64]*u2; 
    temp3 =  T[0 ]*du2 + T[32]*dv2 - T[64]*dv1 + dT[0 ]*u2 + dT[32]*v2 - dT[64]*v1;
    dv2   = -T[32]*du2 + T[0 ]*dv2 + T[64]*du1 - dT[32]*u2 + dT[0 ]*v2 + dT[64]*u1;
    du2   =  temp3;
    if (cond)
    {
        du1 = temp1;
        dv1 = temp2;
    }
    // u_i -> T_{ij} u_j
    matmult_sym(T, u1, v1, u2, v2, temp1, temp2, temp3, cond);
}
__device__ __inline__ void matmult_d_orth(const float *T, const float *dth, float &u1, float &u2, 
                                          float &du1, float &du2, float &temp, bool cond)
{
    temp =  T[0 ]*du1 + T[32]*du2 + dth[0]*(-T[32]*u1 + T[0 ]*u2);
    du2  = -T[32]*du1 + T[0 ]*du2 + dth[0]*(-T[ 0]*u1 - T[32]*u2);
    if (cond)
        du1 = temp;
    matmult_orth(T, u1, u2, temp, cond);
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
    // dJ/du_i^* -> (T^*)_{ij} dJ/du_j = (T_{ji}) dJ/du_j^*
    temp  = conj(T[0 ])*dJdu1 + conj(T[64])*dJdu2;
    dJdu2 = conj(T[32])*dJdu1 + conj(T[96])*dJdu2;
    if (cond)
        dJdu1 = temp;
}
__device__ __inline__ void matmult_bk_sym(const float *T, float *dT, float &u1, float &v1, float &u2, float &v2, 
                                          float &dJdu1, float &dJdv1, float &dJdu2, float &dJdv2, 
                                          float &temp1, float &temp2, float &temp3, bool cond)
{
    // u -> (T^dag) u
    temp1 =  T[0 ]*u1 + T[32]*v1 + T[64]*v2;
    temp2 = -T[32]*u1 + T[0 ]*v1 - T[64]*u2;
    temp3 =  T[0 ]*u2 - T[32]*v2 + T[64]*v1;
    v2    =  T[32]*u2 + T[0 ]*v2 - T[64]*u1;
    u2    =  temp3;
    if (cond)
    {
        u1 = temp1;
        v1 = temp2;
    }
    // dJ/dT = (dJ/du*)* u^T
    atomicAdd(&dT[0 ],  u1*dJdu1 + v1*dJdv1 + u2*dJdu2 + v2*dJdv2);
    atomicAdd(&dT[32], -u1*dJdv1 + v1*dJdu1 + u2*dJdv2 - v2*dJdu2);
    atomicAdd(&dT[64], -u1*dJdv2 + v1*dJdu2 - u2*dJdv1 + v2*dJdu1);
    // dJ/du* -> (T^dag) dJ/du*
    temp1 =  T[0 ]*dJdu1 + T[32]*dJdv1 + T[64]*dJdv2;
    temp2 = -T[32]*dJdu1 + T[0 ]*dJdv1 - T[64]*dJdu2;
    temp3 =  T[0 ]*dJdu2 - T[32]*dJdv2 + T[64]*dJdv1;
    dJdv2 =  T[32]*dJdu2 + T[0 ]*dJdv2 - T[64]*dJdu1;
    dJdu2 =  temp3;
    if (cond)
    {
        dJdu1 = temp1;
        dJdv1 = temp2;
    }
}
__device__ __inline__ void matmult_bk_orth(const float *T, float *d_th, float &u1, float &u2,
                                           float &dJdu1, float &dJdu2, float &temp, bool cond)
{
    temp = T[0 ]*u1 - T[32]*u2;
    u2   = T[32]*u1 + T[0 ]*u2;
    if (cond)
        u1 = temp;
    atomicAdd(d_th, T[0]*(u2*dJdu1 - u1*dJdu2) - T[32]*(u1*dJdu1 + u2*dJdu2));
    temp  = T[0 ]*dJdu1 - T[32]*dJdu2;
    dJdu2 = T[32]*dJdu1 + T[0 ]*dJdu2;
    if (cond)
        dJdu1 = temp;
}