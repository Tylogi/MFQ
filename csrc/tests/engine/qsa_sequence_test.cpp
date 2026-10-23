#include "qsa_sequence.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace mfq::cache;

int main() {
    QsaKvStoreConfig config;
    config.budget_bytes=config.buffer_bytes=4096;
    auto store=std::make_shared<QsaKvStore>(config);
    std::vector<std::uint8_t> rows(19*4);
    for(std::size_t i=0;i<rows.size();++i)rows[i]=static_cast<std::uint8_t>(i);
    QsaKvSequenceSnapshot checkpoint;
    {
        QsaKvSequence sequence(store,4,4);
        sequence.append(rows.data(),7);
        sequence.append(rows.data()+28,12);
        sequence.seal_tail();
        checkpoint=sequence.snapshot();
        store->flush();
        assert(checkpoint.position==19 && store->stats().written_bytes>0);
        sequence.trim(5);
        sequence.append(rows.data()+20,3);
        std::array<std::uint8_t,32> actual{};
        sequence.view().read_rows(0,8,actual.data());
        assert(std::equal(actual.begin(),actual.end(),rows.begin()));
        sequence.restore(checkpoint);
        assert(sequence.position()==19);
        auto invalid=checkpoint;
        ++invalid.position;
        bool rejected=false;
        try {sequence.restore(invalid);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected && sequence.position()==19);
        rejected=false;
        try {sequence.append(nullptr,1);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected && sequence.position()==19);
    }
    // Checkpoints retain both their immutable microblocks and the physical store.
    store.reset();
    std::vector<std::uint8_t> actual(rows.size());
    checkpoint.read_rows(0,19,actual.data());
    assert(actual==rows);
    auto fork=checkpoint.prefix(9);
    QsaKvSequence branch(fork.store,4,4);
    branch.restore(fork);
    branch.append(rows.data()+36,2);
    actual.resize(44);
    branch.view().read_rows(0,11,actual.data());
    assert(std::equal(actual.begin(),actual.end(),rows.begin()));
    bool rejected=false;
    try {checkpoint.read_rows(19,1,actual.data());}catch(const std::out_of_range&){rejected=true;}
    assert(rejected);
    rejected=false;
    try {(void)checkpoint.prefix(20);}catch(const std::out_of_range&){rejected=true;}
    assert(rejected);
    auto foreign=std::make_shared<QsaKvStore>(config);
    QsaKvSequence other(foreign,4,4);
    rejected=false;
    try {other.restore(checkpoint);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected && other.position()==0);
}
