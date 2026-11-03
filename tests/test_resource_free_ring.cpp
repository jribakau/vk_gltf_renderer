// nvapp::ResourceFreeRing: a closure that frees a resource runs once every frame submitted before it
// was queued has completed, whether it was queued in the loop's update step (the previous frame still
// in flight) or while a frame was being recorded.
//
// The loop here orders its steps as Application::headlessRun() and run() do: onUpdate, the wait for
// the frame that last used the ring slot (N frames back), the slot's queue, record, submit, advance.

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
    k++;           // submitted; the next slot
  }
};

}  // namespace

TEST(ResourceFreeRing, QueuedInAnUpdateRunsOnceThePreviousFrameHasCompleted)
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
      // not before frame queuedAt - 1, the one in flight as it was queued, has completed...
      EXPECT_GE(r.completedThrough, int(r.queuedAt) - 1) << slots << " slots, queued in " << r.queuedAt;
      // ...and at the first iteration that has waited for it (the first update has no frame to wait for)
      EXPECT_EQ(r.ranAt, r.queuedAt == 0 ? 0 : r.queuedAt + slots - 1) << slots << " slots, queued in " << r.queuedAt;
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

TEST(ResourceFreeRing, QueuedBeforeAnyFrameRunsAtTheFirst)
{
  Loop loop(2);
  loop.queue();
  loop.iterate({});
  ASSERT_EQ(loop.ran.size(), 1u);
  EXPECT_EQ(loop.ran[0].ranAt, 0u);
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

TEST(ResourceFreeRing, AClosureQueuedByOneRunningWaitsItsOwnFrames)
{
  // A retire that retires something else (a backend's queue draining into the app's): the second
  // waits for the frames in flight as it was queued, like any other.
  Loop loop(2);
  int  second = -1;
  for(uint32_t i = 0; i < 3; i++)
    loop.iterate({});
  loop.iterate([&] { loop.ring.submit([&] { loop.ring.submit([&] { second = int(loop.k); }); }); });  // k = 3
  for(uint32_t i = 0; i < 4; i++)
    loop.iterate({});
  // the first, queued with frame 2 in flight, ran in iteration 4 (its wait is for frame 2); the second,
  // queued then with frame 3 in flight, in iteration 5
  EXPECT_EQ(second, 5);
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
