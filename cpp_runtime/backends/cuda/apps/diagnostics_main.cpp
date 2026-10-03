namespace mfq::cuda::commands {
int run_diagnostics(int argc, char** argv);
}

int main(int argc, char** argv) {
    return mfq::cuda::commands::run_diagnostics(argc, argv);
}
