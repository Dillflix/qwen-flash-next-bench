// Compile against the patched source, without loading a model or GPU backend.
#include "llama-ple-state.h"
#include "server-cache-checkpoint.h"
#include <cassert>
#include <cstring>
#include <list>
#include <vector>

struct history {
    int32_t first_pos = -1, next_pos = -1;
    std::vector<int32_t> toks;
};
struct bytes {
    std::vector<unsigned char> data;
    size_t pos = 0;
    void write(const void * p, size_t n) {
        auto src = static_cast<const unsigned char *>(p);
        data.insert(data.end(), src, src + n);
    }
    void read(void * p, size_t n) {
        if (n > data.size() - pos) { throw std::runtime_error("truncated"); }
        std::memcpy(p, data.data() + pos, n);
        pos += n;
    }
};
struct checkpoint {
    int n_tokens, pos_min, pos_max;
    std::vector<int> data_tgt{1}, data_dft{1}, data_spec{1};
};
struct prompt {
    struct tokens_t { int pos_next(int lcp) const { return lcp; } } tokens;
    std::list<checkpoint> checkpoints;
};

int main() {
    history a{0, 3, {11, 12, 13}}, b{0, 3, {21, 22, 23}};
    bytes snapshot;
    qwen_ple_state_write(snapshot, a);
    qwen_ple_state_read(snapshot, b, 100);
    assert(b.toks == a.toks && b.first_pos == 0 && b.next_pos == 3);
    // A partial checkpoint must replace the complete later history too.
    b.toks.push_back(99); b.next_pos++;
    snapshot.pos = 0;
    qwen_ple_state_read(snapshot, b, 100);
    assert(b.toks == a.toks && b.next_pos == 3);
    for (int corruption = 0; corruption < 4; ++corruption) {
        bytes bad = snapshot; bad.pos = 0;
        if (corruption == 0) { bad.data[0] ^= 1; }
        if (corruption == 1) { bad.data.pop_back(); }
        if (corruption == 2) { bad.data[8] = 4; } // next_pos/count disagree
        bool rejected = false;
        try { qwen_ple_state_read(bad, b, corruption == 3 ? 2 : 100); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected && b.toks == a.toks && b.next_pos == 3);
    }
    bytes empty; history blank;
    qwen_ple_state_write(empty, blank);
    qwen_ple_state_read(empty, b, 100);
    assert(b.toks.empty() && b.next_pos == -1);

    prompt p{{}, {{1225, 1224, 1224}, {2761, 2760, 2760}}};
    assert(server_cache_checkpoint_prefix(p, 2740, true) == 1225);
    assert(server_cache_checkpoint_prefix(p, 2761, true) == 2761);
    assert(server_cache_checkpoint_prefix(p, 0, true) == 0);
    assert(!server_cache_checkpoint_usable(p.checkpoints.back(), 2761, 2760, true));
    p.checkpoints.front().data_dft.clear();
    assert(server_cache_checkpoint_prefix(p, 2740, true) == 0);
    assert(server_cache_checkpoint_prefix(p, 2740, false) == 1225);
    p.checkpoints.front().data_spec.clear();
    assert(server_cache_checkpoint_prefix(p, 2740, false) == 0);
    p.checkpoints.front().data_spec = {1};
    p.checkpoints.front().data_tgt.clear();
    assert(server_cache_checkpoint_prefix(p, 2740, false) == 0);
    p.checkpoints.front().data_tgt = {1};
    p.checkpoints.front().n_tokens = 2741;
    assert(server_cache_checkpoint_prefix(p, 2740, false) == 0);
}
