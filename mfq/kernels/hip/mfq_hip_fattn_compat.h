#pragma once

#include "mfq_hip_compat.h"

using int8x4_t = int8_t __attribute__((ext_vector_type(4)));

#ifndef GGML_USE_HIP
#define GGML_USE_HIP
#endif

#if defined(__gfx908__)
#define CDNA1
#elif defined(__gfx90a__)
#define CDNA2
#elif defined(__gfx942__)
#define CDNA3
#elif defined(__gfx950__)
#define CDNA4
#endif

#if defined(CDNA1) || defined(CDNA2) || defined(CDNA3) || defined(CDNA4)
#define CDNA
#endif

#if defined(__GFX12__)
#define RDNA4
#elif defined(__GFX11__)
#define RDNA3
#endif

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define RDNA3_5
#endif

#if defined(RDNA3) && !defined(RDNA3_5)
#define RDNA3_0
#endif
