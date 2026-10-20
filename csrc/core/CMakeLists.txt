add_library(mfq-core INTERFACE)
add_library(mfq::core ALIAS mfq-core)
target_link_libraries(mfq-core INTERFACE
    mfq-formats mfq-quant-readers mfq-host-parallel mfq-model-graph
    mfq-model-source mfq-json mfq-qsa-kv-store mfq-prefix-blocks)
