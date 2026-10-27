/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Native presentation on Windows (frame_present.hpp): a D3D11 device on the renderer's own GPU
// (matched by LUID), a swapchain on the app's composition surface, and the images and fence it
// shares with Vulkan:
//   - slots: D3D11 textures with NT handles, imported into Vulkan as images (dedicated memory)
//   - ready: a shared D3D11 fence, imported into Vulkan as a timeline semaphore; Vulkan signals a
//     frame's value once its copy is in the slot, D3D's queue waits for it
//   - done: D3D's own fence, signalled after the slot's copy into the swapchain; the CPU reads it
//     to know a slot is free (Vulkan never waits on it on the GPU)

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <volk.h>

#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

#include "frame_present.hpp"

#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr int         kSlots     = 3;
constexpr DXGI_FORMAT kFormat    = DXGI_FORMAT_R8G8B8A8_UNORM;  // the tonemapped image's, sRGB-encoded values
constexpr VkFormat    kVkFormat  = VK_FORMAT_R8G8B8A8_UNORM;
constexpr UINT        kSwapFlags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

std::string failed(const char* what, HRESULT hr)
{
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s failed (0x%08lX)", what, static_cast<unsigned long>(hr));
  return buf;
}

std::string failed(const char* what, VkResult r)
{
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s failed (VkResult %d)", what, int(r));
  return buf;
}

class D3D11Target : public PresentTarget
{
public:
  explicit D3D11Target(const PresentTargetInfo& info)
      : m_device(info.device)
      , m_physicalDevice(info.physicalDevice)
      , m_surface(reinterpret_cast<HANDLE>(info.surface))
  {
  }
  ~D3D11Target() override;

  std::string init(PresentSize frame);

  int         slotCount() const override { return int(m_slots.size()); }
  VkImage     slotImage(int slot) const override { return m_slots[size_t(slot)].image; }
  PresentSize capacity() const override { return m_capacity; }
  VkSemaphore ready() const override { return m_readySemaphore; }
  uint64_t    completed() override { return m_done ? m_done->GetCompletedValue() : 0; }
  std::string growSlots(PresentSize frame) override;
  std::string present(int slot, uint64_t value, PresentSize frame, PresentSize view) override;
  bool        presentable() override;
  bool        waitPresentable(uint32_t timeoutMs) override;

private:
  struct Slot
  {
    ComPtr<ID3D11Texture2D> texture;
    VkImage                 image{};
    VkDeviceMemory          memory{};
  };

  std::string supported() const;
  std::string makeSlots(PresentSize capacity);
  void        destroySlots();
  // D3D's queue is done with every frame handed to it (or the timeout passed)
  bool drain(DWORD timeoutMs);

  VkDevice         m_device{};
  VkPhysicalDevice m_physicalDevice{};
  HANDLE           m_surface{};  // ours: the app duplicated it into this process

  ComPtr<IDXGIFactory4>        m_factory;
  ComPtr<ID3D11Device5>        m_d3d;
  ComPtr<ID3D11DeviceContext4> m_context;
  ComPtr<IDXGISwapChain2>      m_swapchain;
  HANDLE                       m_waitable{};

  ComPtr<ID3D11Fence> m_ready;
  VkSemaphore         m_readySemaphore{};
  ComPtr<ID3D11Fence> m_done;
  HANDLE              m_doneEvent{};
  uint64_t            m_signalled{0};  // the last value D3D's queue was told to signal on m_done

  std::vector<Slot> m_slots;
  PresentSize       m_capacity;
  PresentSize       m_buffer;  // the swapchain's buffers
  PresentSize       m_source;  // the part of them shown, stretched over the whole (SetSourceSize)
};

D3D11Target::~D3D11Target()
{
  // the caller has the Vulkan device idle; D3D's queue may still copy from a slot
  drain(1000);
  if(m_context)
  {
    m_context->ClearState();
    m_context->Flush();
  }
  destroySlots();
  if(m_readySemaphore)
    vkDestroySemaphore(m_device, m_readySemaphore, nullptr);
  m_swapchain.Reset();
  if(m_waitable)
    CloseHandle(m_waitable);
  if(m_doneEvent)
    CloseHandle(m_doneEvent);
  if(m_surface)
    CloseHandle(m_surface);
}

std::string D3D11Target::supported() const
{
  if(!vkGetMemoryWin32HandlePropertiesKHR || !vkImportSemaphoreWin32HandleKHR)
    return "the Vulkan device lacks VK_KHR_external_memory_win32 or VK_KHR_external_semaphore_win32";

  VkPhysicalDeviceExternalImageFormatInfo external{
      .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
  };
  const VkPhysicalDeviceImageFormatInfo2 format{
      .sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .pNext  = &external,
      .format = kVkFormat,
      .type   = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
  };
  VkExternalImageFormatProperties externalProps{.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
  VkImageFormatProperties2        props{.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, .pNext = &externalProps};
  if(vkGetPhysicalDeviceImageFormatProperties2(m_physicalDevice, &format, &props) != VK_SUCCESS
     || !(externalProps.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
    return "the GPU can't import D3D11 textures into Vulkan";

  VkSemaphoreTypeCreateInfo timeline{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
  const VkPhysicalDeviceExternalSemaphoreInfo semaphore{
      .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
      .pNext      = &timeline,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
  };
  VkExternalSemaphoreProperties semaphoreProps{.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
  vkGetPhysicalDeviceExternalSemaphoreProperties(m_physicalDevice, &semaphore, &semaphoreProps);
  if(!(semaphoreProps.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT))
    return "the GPU can't import a D3D fence as a Vulkan timeline semaphore";
  return "";
}

std::string D3D11Target::init(PresentSize frame)
{
  if(std::string err = supported(); !err.empty())
    return err;

  // D3D11 on the renderer's GPU: a laptop's other GPU could not open its images
  VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2  props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &ids};
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &props);
  if(!ids.deviceLUIDValid)
    return "the Vulkan device has no LUID to find its D3D adapter by";
  LUID luid{};
  std::memcpy(&luid, ids.deviceLUID, sizeof luid);

  HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&m_factory));
  if(FAILED(hr))
    return failed("CreateDXGIFactory2", hr);
  ComPtr<IDXGIAdapter1> adapter;
  hr = m_factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
  if(FAILED(hr))
    return failed("EnumAdapterByLuid (the renderer's GPU)", hr);
  const D3D_FEATURE_LEVEL     levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  ComPtr<ID3D11Device>        device;
  ComPtr<ID3D11DeviceContext> context;
  hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
                         UINT(std::size(levels)), D3D11_SDK_VERSION, &device, nullptr, &context);
  if(FAILED(hr))
    return failed("D3D11CreateDevice", hr);
  if(FAILED(hr = device.As(&m_d3d)) || FAILED(hr = context.As(&m_context)))
    return failed("ID3D11Device5 (fences)", hr);

  // ready: D3D's fence, shared, imported as Vulkan's timeline semaphore
  hr = m_d3d->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_ready));
  if(FAILED(hr))
    return failed("CreateFence (shared)", hr);
  HANDLE readyHandle{};
  hr = m_ready->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &readyHandle);
  if(FAILED(hr))
    return failed("ID3D11Fence::CreateSharedHandle", hr);
  VkSemaphoreTypeCreateInfo timeline{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
  const VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timeline};
  VkResult                    vr = vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_readySemaphore);
  if(vr == VK_SUCCESS)
  {
    const VkImportSemaphoreWin32HandleInfoKHR import{
        .sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
        .semaphore  = m_readySemaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
        .handle     = readyHandle,
    };
    vr = vkImportSemaphoreWin32HandleKHR(m_device, &import);
  }
  // an NT handle's import doesn't take it over
  CloseHandle(readyHandle);
  if(vr != VK_SUCCESS)
    return failed("importing the D3D fence into Vulkan", vr);

  hr = m_d3d->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_done));
  if(FAILED(hr))
    return failed("CreateFence", hr);
  m_doneEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

  if(std::string err = makeSlots(frame); !err.empty())
    return err;

  // the swapchain, on the app's surface
  ComPtr<IDXGIFactoryMedia> media;
  if(FAILED(hr = m_factory.As(&media)))
    return failed("IDXGIFactoryMedia", hr);
  const DXGI_SWAP_CHAIN_DESC1 desc{
      .Width       = frame.width,
      .Height      = frame.height,
      .Format      = kFormat,
      .SampleDesc  = {1, 0},
      .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
      .BufferCount = 2,
      .Scaling     = DXGI_SCALING_STRETCH,
      .SwapEffect  = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
      .AlphaMode   = DXGI_ALPHA_MODE_IGNORE,  // the tonemapped image's alpha is not coverage
      .Flags       = kSwapFlags,
  };
  ComPtr<IDXGISwapChain1> swapchain;
  hr = media->CreateSwapChainForCompositionSurfaceHandle(m_d3d.Get(), m_surface, &desc, nullptr, &swapchain);
  if(FAILED(hr))
    return failed("CreateSwapChainForCompositionSurfaceHandle", hr);
  if(FAILED(hr = swapchain.As(&m_swapchain)))
    return failed("IDXGISwapChain2", hr);
  m_swapchain->SetMaximumFrameLatency(1);
  m_waitable = m_swapchain->GetFrameLatencyWaitableObject();
  if(ComPtr<IDXGISwapChain3> sc3; SUCCEEDED(m_swapchain.As(&sc3)))
    sc3->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
  m_buffer = m_source = frame;
  return "";
}

std::string D3D11Target::makeSlots(PresentSize capacity)
{
  m_capacity = capacity;
  m_slots.resize(kSlots);
  for(Slot& slot : m_slots)
  {
    const D3D11_TEXTURE2D_DESC desc{
        .Width      = capacity.width,
        .Height     = capacity.height,
        .MipLevels  = 1,
        .ArraySize  = 1,
        .Format     = kFormat,
        .SampleDesc = {1, 0},
        .Usage      = D3D11_USAGE_DEFAULT,
        .BindFlags  = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
        .MiscFlags  = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
    };
    HRESULT hr = m_d3d->CreateTexture2D(&desc, nullptr, &slot.texture);
    if(FAILED(hr))
      return failed("CreateTexture2D (shared)", hr);
    ComPtr<IDXGIResource1> resource;
    HANDLE                 handle{};
    if(FAILED(hr = slot.texture.As(&resource))
       || FAILED(hr = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &handle)))
      return failed("IDXGIResource1::CreateSharedHandle", hr);

    VkExternalMemoryImageCreateInfo external{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
    };
    const VkImageCreateInfo imageInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = &external,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = kVkFormat,
        .extent        = {capacity.width, capacity.height, 1},
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkResult vr = vkCreateImage(m_device, &imageInfo, nullptr, &slot.image);
    if(vr != VK_SUCCESS)
    {
      CloseHandle(handle);
      return failed("vkCreateImage (shared)", vr);
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_device, slot.image, &req);
    VkMemoryWin32HandlePropertiesKHR handleProps{.sType = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
    vkGetMemoryWin32HandlePropertiesKHR(m_device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, handle, &handleProps);
    uint32_t bits = req.memoryTypeBits & (handleProps.memoryTypeBits ? handleProps.memoryTypeBits : ~0u);
    if(bits == 0)
    {
      CloseHandle(handle);
      return "no Vulkan memory type takes the shared D3D11 texture";
    }
    uint32_t typeIndex = 0;
    while(!(bits & (1u << typeIndex)))
      typeIndex++;
    VkImportMemoryWin32HandleInfoKHR import{
        .sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
        .handle     = handle,
    };
    VkMemoryDedicatedAllocateInfo dedicated{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &import, .image = slot.image};
    const VkMemoryAllocateInfo alloc{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &dedicated,
        .allocationSize  = req.size,
        .memoryTypeIndex = typeIndex,
    };
    vr = vkAllocateMemory(m_device, &alloc, nullptr, &slot.memory);
    // an NT handle's import doesn't take it over
    CloseHandle(handle);
    if(vr != VK_SUCCESS)
      return failed("importing the shared D3D11 texture into Vulkan", vr);
    if((vr = vkBindImageMemory(m_device, slot.image, slot.memory, 0)) != VK_SUCCESS)
      return failed("vkBindImageMemory (shared)", vr);
  }
  return "";
}

void D3D11Target::destroySlots()
{
  for(Slot& slot : m_slots)
  {
    if(slot.image)
      vkDestroyImage(m_device, slot.image, nullptr);
    if(slot.memory)
      vkFreeMemory(m_device, slot.memory, nullptr);
  }
  m_slots.clear();
}

bool D3D11Target::drain(DWORD timeoutMs)
{
  if(!m_done || m_done->GetCompletedValue() >= m_signalled)
    return true;
  m_context->Flush();
  if(FAILED(m_done->SetEventOnCompletion(m_signalled, m_doneEvent)))
    return false;
  return WaitForSingleObject(m_doneEvent, timeoutMs) == WAIT_OBJECT_0;
}

std::string D3D11Target::growSlots(PresentSize frame)
{
  if(fits(m_capacity, frame))
    return "";
  if(!drain(1000))
    return "D3D did not finish with the shared images";
  m_context->ClearState();
  m_context->Flush();
  destroySlots();
  return makeSlots(growCapacity(m_capacity, frame));
}

std::string D3D11Target::present(int slot, uint64_t value, PresentSize frame, PresentSize view)
{
  // The swapchain is the viewport's size, and the frame is its source region: a composition
  // swapchain stretches that over its whole buffer, so a frame capped by Max resolution, or one
  // still at its old size while a resize is on its way, fills the viewport in shape. A frame larger
  // than the viewport (an export) takes a swapchain its own size and shows 1:1.
  HRESULT           hr;
  const PresentSize buffer = view.width > 0 && view.height > 0 && fits(view, frame) ? view : frame;
  if(!(buffer == m_buffer))
  {
    if(!drain(1000))
      return "D3D did not finish with the swapchain";
    m_context->ClearState();
    m_context->Flush();
    if(FAILED(hr = m_swapchain->ResizeBuffers(0, buffer.width, buffer.height, DXGI_FORMAT_UNKNOWN, kSwapFlags)))
      return failed("ResizeBuffers", hr);
    m_buffer = m_source = buffer;
  }
  // D3D's queue waits for Vulkan's copy into the slot; the CPU doesn't
  m_context->Wait(m_ready.Get(), value);
  ComPtr<ID3D11Texture2D> back;
  if(FAILED(hr = m_swapchain->GetBuffer(0, IID_PPV_ARGS(&back))))
    return failed("IDXGISwapChain::GetBuffer", hr);
  const D3D11_BOX box{0, 0, 0, frame.width, frame.height, 1};
  m_context->CopySubresourceRegion(back.Get(), 0, 0, 0, 0, m_slots[size_t(slot)].texture.Get(), 0, &box);
  if(!(frame == m_source))
  {
    if(FAILED(hr = m_swapchain->SetSourceSize(frame.width, frame.height)))
      return failed("SetSourceSize", hr);
    m_source = frame;
  }
  hr = m_swapchain->Present(1, 0);
  // the slot is free once the copy above has run, whatever became of the present
  m_context->Signal(m_done.Get(), value);
  m_signalled = value;
  m_context->Flush();
  if(FAILED(hr))
  {
    if(hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET)
      return failed("Present (device removed)", m_d3d->GetDeviceRemovedReason());
    return failed("Present", hr);
  }
  return "";
}

bool D3D11Target::presentable()
{
  return waitPresentable(0);
}

bool D3D11Target::waitPresentable(uint32_t timeoutMs)
{
  return WaitForSingleObjectEx(m_waitable, timeoutMs, FALSE) == WAIT_OBJECT_0;
}

}  // namespace

std::unique_ptr<PresentTarget> createPresentTarget(const PresentTargetInfo& info, std::string& error)
{
  // owns the surface handle from here: a target that fails to start closes it
  auto target = std::make_unique<D3D11Target>(info);
  error       = target->init(info.frame);
  if(!error.empty())
    return nullptr;
  return target;
}

#endif  // _WIN32
