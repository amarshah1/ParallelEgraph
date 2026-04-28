#pragma once
// FxHash — the hash rustc uses (from rustc-hash crate).
// We need bit-for-bit equivalence with the Rust side so that any cross-
// validation debugging that prints sig_hash values can diff raw.
//
// FxHasher state transition (for each input chunk), taken from rustc-hash:
//   state = (state.rotate_left(5) ^ input) * 0x517c_c1b7_2722_0a95
// Finalize is just `state`. FxHasher::default() starts with state = 0.
//
// Rust's Hash impl for integers writes them little-endian through write_u64
// (for u64) or write_u32 (for u32). Strings hash byte-by-byte (but rustc-hash
// actually chunks them into u64s). For our EGraph use we call:
//   - str.hash(&mut h)     — which writes the bytes + length
//   - u32.hash(&mut h)     — which calls write_u32
// We model exactly those two paths.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string_view>

namespace pe {

class FxHasher {
 public:
  FxHasher() = default;
  explicit FxHasher(std::uint64_t seed) : state_(seed) {}

  // Mirror rustc-hash's single-chunk mixing step.
  inline void write_u64(std::uint64_t word) {
    // rotate_left(5)
    std::uint64_t rot = (state_ << 5) | (state_ >> 59);
    state_ = (rot ^ word) * 0x517C'C1B7'2722'0A95ULL;
  }

  inline void write_u32(std::uint32_t word) {
    // rustc-hash promotes the u32 to a 64-bit chunk at the low bits.
    write_u64(static_cast<std::uint64_t>(word));
  }

  // rustc's `<str as Hash>::hash` writes the bytes then writes a trailing
  // 0xFF byte (the "str stream-end marker"). rustc-hash batches the bytes
  // into u64 chunks; the tail is handled separately. We replicate that.
  void write_bytes(const std::uint8_t* data, std::size_t len) {
    while (len >= 8) {
      std::uint64_t word;
      std::memcpy(&word, data, 8);
      write_u64(word);
      data += 8;
      len -= 8;
    }
    if (len > 0) {
      std::uint64_t tail = 0;
      std::memcpy(&tail, data, len);
      write_u64(tail);
    }
  }

  void write_str(std::string_view s) {
    // `<str as Hash>::hash` writes bytes then writes a 0xFF terminator byte.
    write_bytes(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
    write_u8(0xFF);
  }

  inline void write_u8(std::uint8_t b) {
    write_u64(static_cast<std::uint64_t>(b));
  }

  inline std::uint64_t finish() const { return state_; }

 private:
  std::uint64_t state_ = 0;
};

}  // namespace pe
