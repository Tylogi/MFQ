#pragma once

// Copy the aligned body with wide loads even when the field has a short tail.
// Compact expert fields can have an eight-byte slot stride; requiring a
// sixteen-byte total length made those whole fields fall back to byte loads.
template<class Vector>
__device__ __forceinline__ void mfq_moe_copy_vector_body(
        unsigned char* destination, const unsigned char* source,
        unsigned long long bytes, unsigned long long worker,
        unsigned long long workers) {
    const auto count=bytes/sizeof(Vector);
    auto* output=reinterpret_cast<Vector*>(destination);
    const auto* input=reinterpret_cast<const Vector*>(source);
    for(auto index=worker;index<count;index+=workers)output[index]=input[index];
    for(auto index=count*sizeof(Vector)+worker;index<bytes;index+=workers)
        destination[index]=source[index];
}

__device__ __forceinline__ void mfq_moe_copy_bytes(
        unsigned char* destination, const unsigned char* source,
        unsigned long long bytes, unsigned long long worker,
        unsigned long long workers) {
    const auto alignment=reinterpret_cast<unsigned long long>(destination) |
                         reinterpret_cast<unsigned long long>(source);
    if(!(alignment&15))mfq_moe_copy_vector_body<uint4>(destination,source,bytes,worker,workers);
    else if(!(alignment&7))mfq_moe_copy_vector_body<uint2>(destination,source,bytes,worker,workers);
    else if(!(alignment&3))mfq_moe_copy_vector_body<unsigned int>(destination,source,bytes,worker,workers);
    else if(!(alignment&1))mfq_moe_copy_vector_body<unsigned short>(destination,source,bytes,worker,workers);
    else mfq_moe_copy_vector_body<unsigned char>(destination,source,bytes,worker,workers);
}
