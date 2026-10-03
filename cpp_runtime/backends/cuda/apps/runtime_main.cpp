namespace mfq::cuda::commands {
int run_runtime(int argc, char** argv);
}

int main(int argc, char** argv) {
    return mfq::cuda::commands::run_runtime(argc, argv);
}
