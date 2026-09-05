// Synthesizes a structurally valid .isf save of an arbitrary digit length, for use as a fixed-length throughput probe.
//
//   mkisf <digits> <iteration> <out.isf>
//
// The digits are pseudo-random, so the file is NOT a real 196 trajectory - it exists only to put the packed core at a
// chosen length. That is legitimate for a throughput measurement because the core's cost is data-independent: the SIMD
// passes are branch-free and touch every limb exactly once regardless of digit values, and the only value-dependent
// control flow (the parity path) is selected by the digit COUNT, not the digits.
//
// The file still has to satisfy load_result, which enforces a CRC-16/CCITT over the body and ISFTOOL's mod-9 residue
// rule: for initial value 196 (residue 7), the stored digit sum mod 9 must equal 7 * 2^(iteration % 6). The iteration
// is therefore chosen a multiple of 6 so the target residue is 7.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static uint16_t crc16_table[256];

static void crc16_init_table()
{
  for (uint32_t i = 0; i < 256; ++i)
  {
    uint16_t c = (uint16_t)(i << 8);
    for (int b = 0; b < 8; ++b)
    {
      c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    crc16_table[i] = c;
  }
}

static uint16_t crc16_update(uint16_t crc, const uint8_t *data, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    crc = (uint16_t)((crc << 8) ^ crc16_table[(crc >> 8) ^ data[i]]);
  }
  return crc;
}

static const size_t kDigitsPerLine = 70;

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    std::fprintf(stderr, "usage: mkisf <digits> <iteration> <out.isf>\n");
    return 2;
  }

  const size_t len = (size_t)strtoull(argv[1], nullptr, 10);
  const uint64_t iteration = strtoull(argv[2], nullptr, 10);
  const char *path = argv[3];

  if (len < 2)
  {
    std::fprintf(stderr, "digits must be >= 2\n");
    return 2;
  }

  std::vector<uint8_t> d(len);
  uint64_t s = 0x9E3779B97F4A7C15ull ^ len;
  for (size_t i = 0; i < len; ++i)
  {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    d[i] = (uint8_t)(s % 10);
  }
  if (d[0] == 0)
  {
    d[0] = 1;
  }

  int csum = 0;
  for (size_t i = 0; i < len; ++i)
  {
    csum = (csum + d[i]) % 9;
  }

  int target = 7;
  for (uint64_t i = 0; i < (iteration % 6); ++i)
  {
    target = (target * 2) % 9;
  }
  if (target == 0 || target == 3 || target == 6)
  {
    std::fprintf(stderr, "iteration %llu yields an unreachable residue\n", (unsigned long long)iteration);
    return 2;
  }

  const int delta = ((target - csum) % 9 + 9) % 9;
  d[len - 1] = (uint8_t)((d[len - 1] + delta) % 9);

  std::string body;
  body.reserve(len + 2 * (len / kDigitsPerLine) + 128);
  body += "Iteration:        " + std::to_string(iteration) + "\n";
  body += "Number of digits: " + std::to_string(len) + "\n";
  for (size_t i = 0; i < len; i += kDigitsPerLine)
  {
    const size_t n = (kDigitsPerLine < len - i) ? kDigitsPerLine : len - i;
    for (size_t j = 0; j < n; ++j)
    {
      body.push_back((char)('0' + d[i + j]));
    }
    body += "\n";
  }

  crc16_init_table();
  const uint16_t crc = crc16_update(0xFFFF, (const uint8_t *)body.data(), body.size());

  FILE *f = nullptr;
  if (fopen_s(&f, path, "wb") != 0 || !f)
  {
    std::fprintf(stderr, "cannot open %s\n", path);
    return 1;
  }
  std::fprintf(f, "Automatic save #%u\nInitial value:    196\n", (unsigned)crc);
  std::fwrite(body.data(), 1, body.size(), f);
  std::fclose(f);

  std::printf("wrote %s (%zu digits, iteration %llu)\n", path, len, (unsigned long long)iteration);
  return 0;
}
