#pragma once

// When the path tracer traces its selection ray per pixel into the selection image (ePtWriteSelection).
// The image's readers are a selection's silhouette and the ImGui hover readback. A still writes it on
// its frame 0. Every DLSS frame is a frame 0, and there the ray is wasted unless one of them reads it.
// A frame 0 that leaves it out leaves it stale, so a reader that appears later (a selection made while
// the view rests on DLSS frames, whose count climbs) gets it written once on the next frame.
//
//   if(selection.due(frameCount == 0, useDlss, selectionRead))
//     flags |= ePtWriteSelection;
struct SelectionWrite
{
  bool stale = true;  // the image holds no current object IDs

  bool due(bool firstFrame, bool dlssFrame, bool read)
  {
    const bool write = firstFrame ? (!dlssFrame || read) : (read && stale);
    if(firstFrame || write)
      stale = !write;
    return write;
  }
};
