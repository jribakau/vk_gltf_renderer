// The figure's thread pool: parallelFor covers every index once, from any thread, a pool job included.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include <figure/jobs/memory_gate.hpp>
#include <figure/jobs/pool.hpp>

TEST(Jobs, ParallelForCoversEveryIndexOnce)
{
  std::vector<std::atomic<int>> hits(10007);
  fv::jobs::parallelFor(hits.size(), 64, [&](size_t begin, size_t end) {
    for(size_t i = begin; i < end; i++)
      hits[i]++;
  });
  for(size_t i = 0; i < hits.size(); i++)
    ASSERT_EQ(hits[i].load(), 1) << i;
}

TEST(Jobs, ParallelForInsideAPoolJobFinishes)
{
  // every pool thread busy with a job that itself splits work: the callers take the chunks
  const size_t                   jobs = fv::jobs::pool().size() + 2;
  std::atomic<size_t>            total{0};
  std::vector<std::future<void>> running;
  for(size_t j = 0; j < jobs; j++)
    running.push_back(fv::jobs::pool().submit([&] {
      fv::jobs::parallelFor(1000, 10, [&](size_t begin, size_t end) { total += end - begin; });
    }));
  for(auto& f : running)
    ASSERT_EQ(f.wait_for(std::chrono::seconds(30)), std::future_status::ready);
  EXPECT_EQ(total.load(), jobs * 1000);
}

TEST(Jobs, ParallelForRethrowsAfterEveryChunk)
{
  std::atomic<int> chunks{0};
  EXPECT_THROW(fv::jobs::parallelFor(100, 10,
                                     [&](size_t begin, size_t) {
                                       chunks++;
                                       if(begin == 50)
                                         throw std::runtime_error("chunk 5");
                                     }),
               std::runtime_error);
  EXPECT_EQ(chunks.load(), 10);
}

// Large builds side by side only as far as they fit; one alone always goes, whatever it asks.
TEST(Jobs, TheMemoryGateLetsThroughWhatFitsAndOneAloneAlways)
{
  std::atomic<uint64_t> available{1000};
  fv::jobs::MemoryGate  gate([&] { return available.load(); });
  {
    auto huge = gate.enter(5000);  // alone
    EXPECT_EQ(gate.inside(), 5000u);
  }
  EXPECT_EQ(gate.inside(), 0u);

  auto first = gate.enter(600);
  auto small = gate.enter(300);
  EXPECT_EQ(gate.inside(), 900u);
  std::atomic<bool> through{false};
  std::thread       waiting([&] {
    auto third = gate.enter(600);
    through    = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(through.load());
  small.release();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(through.load());  // 600 + 600 doesn't fit either
  first.release();
  waiting.join();
  EXPECT_TRUE(through.load());
  EXPECT_EQ(gate.inside(), 0u);
}
