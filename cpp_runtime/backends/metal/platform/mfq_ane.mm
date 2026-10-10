#include "mfq_ane.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mfq::metal {
namespace {

Class descriptor, model_class, request_class, surface_class;

void initialize() {
    static std::once_flag once;
    std::call_once(once, [] {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        descriptor = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        model_class = NSClassFromString(@"_ANEInMemoryModel");
        request_class = NSClassFromString(@"_ANERequest");
        surface_class = NSClassFromString(@"_ANEIOSurfaceObject");
    });
    if (!descriptor || !model_class || !request_class || !surface_class)
        throw std::runtime_error("Apple Neural Engine private runtime is unavailable");
    if (![descriptor respondsToSelector:@selector(modelWithMILText:weights:optionsPlist:)] ||
        ![model_class respondsToSelector:@selector(inMemoryModelWithDescriptor:)] ||
        ![request_class respondsToSelector:@selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:)] ||
        ![surface_class respondsToSelector:@selector(objectWithIOSurface:)])
        throw std::runtime_error("Apple Neural Engine private API is incompatible");
}

IOSurfaceRef make_surface(std::size_t size) {
    auto result = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(size), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(size),
        (id)kIOSurfaceAllocSize: @(size), (id)kIOSurfacePixelFormat: @0});
    if (!result) throw std::runtime_error("ANE IOSurface allocation failed");
    return result;
}

std::runtime_error failure(const char* phase, NSError* error) {
    return std::runtime_error(std::string("ANE ") + phase + ": " +
        (error ? error.description.UTF8String : "unsupported private API"));
}

NSString* matmul_program(int rows, int inner, int columns) {
    return [NSString stringWithFormat:@"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"},"
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"},"
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "func main<ios18>(tensor<fp16, [1,%d,1,%d]> x) {\n"
        "tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "tensor<fp16, [1,%d,1,%d]> a = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"a\")];\n"
        "tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n"
        "tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "tensor<fp16, [1,%d,1,%d]> w = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"w\")];\n"
        "tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=a)[name=string(\"a2\")];\n"
        "tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"
        "tensor<fp16, [1,1,%d,%d]> at = transpose(perm=pm,x=a2)[name=string(\"at\")];\n"
        "tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "tensor<fp16, [1,1,%d,%d]> w2 = reshape(shape=rw,x=w)[name=string(\"w2\")];\n"
        "bool f = const()[name=string(\"f\"), val=bool(false)];\n"
        "tensor<fp16, [1,1,%d,%d]> y = matmul(transpose_x=f,transpose_y=f,x=at,y=w2)[name=string(\"y\")];\n"
        "tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=y)[name=string(\"yt\")];\n"
        "tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "tensor<fp16, [1,%d,1,%d]> out = reshape(shape=ro,x=yt)[name=string(\"out\")];\n"
        "} -> (out);\n}\n",
        inner, rows + columns, inner, rows, inner, rows, rows,
        inner, columns, inner, columns, inner, rows, inner, rows, rows, inner,
        inner, columns, inner, columns, rows, columns, columns, rows,
        columns, rows, columns, rows];
}

} // namespace

struct AneMatmul::Impl {
    int rows, inner, columns;
    __strong id model = nil;
    __strong id request = nil;
    __strong NSString* directory = nil;
    IOSurfaceRef input = nullptr, output = nullptr;
    std::mutex mutex;

    ~Impl() {
        @autoreleasepool {
            if (model && [model respondsToSelector:@selector(unloadWithQoS:error:)]) {
                NSError* error = nil;
                ((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
                    model, @selector(unloadWithQoS:error:), 17, &error);
            }
            if (input) CFRelease(input);
            if (output) CFRelease(output);
            if (directory) [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
        }
    }
};

AneMatmul::AneMatmul(int rows, int inner, int columns) : impl_(std::make_unique<Impl>()) {
    if (rows <= 0 || inner <= 0 || columns <= 0 || rows > 8192 || inner > 8192 || columns > 8192)
        throw std::invalid_argument("ANE matmul dimensions must be within 1..8192");
    impl_->rows = rows;
    impl_->inner = inner;
    impl_->columns = columns;
    @autoreleasepool {
        initialize();
        auto program = [matmul_program(rows, inner, columns) stringByReplacingOccurrencesOfString:
            @"name=string(\"out\")" withString:[NSString stringWithFormat:@"name=string(\"out_%@\")", NSUUID.UUID.UUIDString]];
        auto text = [program dataUsingEncoding:NSUTF8StringEncoding];
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(descriptor,
            @selector(modelWithMILText:weights:optionsPlist:), text, @{}, nil);
        if (!desc) throw failure("descriptor", nil);
        impl_->model = ((id(*)(Class, SEL, id))objc_msgSend)(model_class,
            @selector(inMemoryModelWithDescriptor:), desc);
        if (!impl_->model) throw failure("model", nil);
        for (const auto selector : {@selector(hexStringIdentifier), @selector(compileWithQoS:options:error:),
            @selector(loadWithQoS:options:error:), @selector(evaluateWithQoS:options:request:error:)})
            if (![impl_->model respondsToSelector:selector]) throw failure("model API", nil);
        id identity = ((id(*)(id, SEL))objc_msgSend)(impl_->model, @selector(hexStringIdentifier));
        auto invalid = [[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefABCDEF_"] invertedSet];
        if (![identity isKindOfClass:[NSString class]] || [identity length] < 16 || [identity length] > 255 ||
            [identity rangeOfCharacterFromSet:invalid].location != NSNotFound)
            throw failure("identifier", nil);
        impl_->directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identity];
        NSError* error = nil;
        auto files = [NSFileManager defaultManager];
        if (![files createDirectoryAtPath:impl_->directory withIntermediateDirectories:YES
            attributes:@{NSFilePosixPermissions: @0700} error:&error] ||
            ![text writeToFile:[impl_->directory stringByAppendingPathComponent:@"model.mil"] atomically:YES])
            throw failure("staging", error);
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(impl_->model,
            @selector(compileWithQoS:options:error:), 17, @{}, &error)) throw failure("compile", error);
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(impl_->model,
            @selector(loadWithQoS:options:error:), 17, @{}, &error)) throw failure("load", error);
        impl_->input = make_surface(std::size_t(inner) * (rows + columns) * sizeof(_Float16));
        impl_->output = make_surface(std::size_t(columns) * rows * sizeof(_Float16));
        id in = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surface_class,
            @selector(objectWithIOSurface:), impl_->input);
        id out = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surface_class,
            @selector(objectWithIOSurface:), impl_->output);
        if (!in || !out) throw failure("surface binding", nil);
        impl_->request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(request_class,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[in], @[@0], @[out], @[@0], nil, nil, @0);
        if (!impl_->request) throw failure("request", nil);
    }
}

AneMatmul::~AneMatmul() = default;

std::size_t AneMatmul::bytes() const noexcept {
    return std::size_t(impl_->inner) * (impl_->rows + impl_->columns) * sizeof(_Float16) +
        std::size_t(impl_->columns) * impl_->rows * sizeof(_Float16);
}

std::vector<float> AneMatmul::operator()(std::span<const float> left, std::span<const float> right) {
    std::lock_guard lock(impl_->mutex);
    const auto [rows, inner, columns] = std::tuple{impl_->rows, impl_->inner, impl_->columns};
    if (left.size() != std::size_t(rows) * inner || right.size() != std::size_t(inner) * columns)
        throw std::invalid_argument("ANE matmul input shape mismatch");
    for (auto values : {left, right}) for (auto value : values)
        if (!std::isfinite(value) || std::abs(value) > 65504) throw std::invalid_argument("ANE input is not finite FP16");
    @autoreleasepool {
        if (IOSurfaceLock(impl_->input, 0, nullptr) != kIOReturnSuccess) throw failure("input lock", nil);
        auto* data = static_cast<_Float16*>(IOSurfaceGetBaseAddress(impl_->input));
        for (int k = 0; k < inner; ++k) {
            for (int m = 0; m < rows; ++m) data[k * (rows + columns) + m] = left[m * inner + k];
            for (int n = 0; n < columns; ++n) data[k * (rows + columns) + rows + n] = right[k * columns + n];
        }
        IOSurfaceUnlock(impl_->input, 0, nullptr);
        NSError* error = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(impl_->model,
            @selector(evaluateWithQoS:options:request:error:), 17, @{}, impl_->request, &error))
            throw failure("evaluate", error);
        std::vector<float> result(std::size_t(rows) * columns);
        if (IOSurfaceLock(impl_->output, kIOSurfaceLockReadOnly, nullptr) != kIOReturnSuccess)
            throw failure("output lock", nil);
        const auto* output = static_cast<const _Float16*>(IOSurfaceGetBaseAddress(impl_->output));
        for (int m = 0; m < rows; ++m) for (int n = 0; n < columns; ++n)
            result[m * columns + n] = output[n * rows + m];
        IOSurfaceUnlock(impl_->output, kIOSurfaceLockReadOnly, nullptr);
        return result;
    }
}

} // namespace mfq::metal
