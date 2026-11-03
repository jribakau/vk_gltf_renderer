// nvapp::ResourceFreeRing: a closure that frees a resource runs once the frame of the loop
// iteration it was queued in has completed, whether it was queued in the update step (the previous
// frame still in flight, this frame's copies about to be recorded) or while the frame was recorded.
//
// The loop here orders its steps as Application::headlessRun() and run() do: onUpdate, the wait for
// the frame that last used the ring slot (N frames back), the slot's drain, record, submit, advance.

#include <gtest/gtest.h>
#include <nvapp/resource_free_ring.hpp>

#include <algorithm>
#include <functional>
#include <vector>

namespace {

struct Ran
{
  uint32_t queuedAt;          // the iteration the closure was queued in
  uint32_t ranAt;             // the iteration it ran in
  int      completedThrough;  // the latest frame the loop had waited for by then
};

struct Loop
{
  explicit Loop(uint32_t slots)
      : n(slots)
  {
    ring.reset(slots);
  }

  nvapp::ResourceFreeRing ring;
  uint32_t                n;
  uint32_t                k                = 0;   // the iteration, and the frame it records
  int                     completedThrough = -1;  // waited for: nothing yet
  std::vector<Ran>        ran;

  // A closure queued now, saying when it ran and what had completed by then.
  void queue()
  {
    ring.submit([this, at = k] { ran.push_back({at, k, completedThrough}); });
  }

  void iterate(const std::function<void()>& onUpdate, const std::function<void()>& onRender = {})
  {
    if(onUpdate)
      onUpdate();  // frames before k may be in flight
    completedThrough = std::max(completedThrough, int(k) - int(n));  // waitForFrameCompletion()
    ring.frameCompleted(k % n);                                        // freeResourcesQueue()
    if(onRender)
      onRender();  // frame k is recorded
    ring.frameAdvanced();  // submitted; advanceFrame()
    k++;
  }
};

}  // namespace

TEST(ResourceFreeRing, QueuedInAnUpdateRunsOnceThatIterationsFrameHasCompleted)
{
  for(uint32_t slots : {2u, 3u})
  {
    Loop loop(slots);
    for(uint32_t i = 0; i < 12; i++)
      loop.iterate([&] { loop.queue(); });
    for(uint32_t i = 0; i < slots; i++)  // the loop goes on: the last ones drain through their waits
      loop.iterate({});
    ASSERT_EQ(loop.ran.size(), 12u) << slots << " slots";
    for(const Ran& r : loop.ran)
    {
      // not before frame queuedAt has completed: the previous frame, in flight as it was queued,
      // and this one, whose recording may copy from what the closure frees, have both...
      EXPECT_GE(r.completedThrough, int(r.queuedAt)) << slots << " slots, queued in " << r.queuedAt;
      // ...and at the first iteration that has waited for it
      EXPECT_EQ(r.ranAt, r.queuedAt + slots) << slots << " slots, queued in " << r.queuedAt;
    }
  }
}

TEST(ResourceFreeRing, QueuedWhileRecordingRunsOnceThatFrameHasCompleted)
{
  for(uint32_t slots : {2u, 3u})
  {
    Loop loop(slots);
    for(uint32_t i = 0; i < 12; i++)
      loop.iterate({}, [&] { loop.queue(); });
    for(uint32_t i = 0; i < slots; i++)
      loop.iterate({});
    ASSERT_EQ(loop.ran.size(), 12u) << slots << " slots";
    for(const Ran& r : loop.ran)
    {
      EXPECT_GE(r.completedThrough, int(r.queuedAt)) << slots << " slots, queued in " << r.queuedAt;
      EXPECT_EQ(r.ranAt, r.queuedAt + slots) << slots << " slots, queued in " << r.queuedAt;
    }
  }
}

TEST(ResourceFreeRing, QueuedBeforeAnyFrameWaitsForTheFirst)
{
  Loop loop(2);
  loop.queue();
  for(uint32_t i = 0; i < 3; i++)
    loop.iterate({});
  ASSERT_EQ(loop.ran.size(), 1u);
  EXPECT_EQ(loop.ran[0].ranAt, 2u);
  EXPECT_GE(loop.ran[0].completedThrough, 0);
}

TEST(ResourceFreeRing, WithNoSlotsAClosureRunsAsItIsQueued)
{
  nvapp::ResourceFreeRing ring;
  int                     runs = 0;
  ring.submit([&] { runs++; });
  EXPECT_EQ(runs, 1);
  ring.reset(0);
  ring.submit([&] { runs++; });
  EXPECT_EQ(runs, 2);
}

TEST(ResourceFreeRing, AClosureQueuedByOneRunningWaitsItsOwnFrame)
{
  // A retire that retires something else (a backend's queue draining into the app's): the second
  // runs once the frame of the iteration it was queued in has completed, like any other.
  Loop loop(2);
  int  second = -1;
  loop.iterate([&] { loop.ring.submit([&] { loop.ring.submit([&] { second = int(loop.k); }); }); });  // k = 0
  for(uint32_t i = 0; i < 6; i++)
    loop.iterate({});
  // the first ran in iteration 2 (after the wait for frame 0); the second, queued then, in iteration 4
  EXPECT_EQ(second, 4);
}

TEST(ResourceFreeRing, ResetRunsEverythingIncludingWhatAClosureQueues)
{
  nvapp::ResourceFreeRing ring;
  ring.reset(2);
  int runs = 0;
  ring.submit([&] {
    runs++;
    ring.submit([&] { runs++; });
  });
  ring.reset(0);
  EXPECT_EQ(runs, 2);
}
