use crate::nodes::{
    blur::Blur, color_change::ColorChange, concat::Concat, crop::Crop, fps_change::FpsChange, load::Load, output::Output, pancake::Pancake, saturation_change::SaturationChange, split::Split, transform::Transform, transition::Transition, trim::Trim,
};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
use ffmpeg_next::Rational;
use std::sync::Arc;
pub type HashOutput = u64;
/// Planar YUV 4:2:2, tightly packed (no stride padding).
/// `y` is `width * height`; `u` and `v` are each `(width / 2) * height`.
/// `a` is an optional straight (not premultiplied) alpha plane of `width * height`:
/// 0 is transparent, 255 is opaque, and `None` means the whole frame is opaque.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub a: Option<Vec<u8>>,
}
/// One plane of a frame whose GPU work may still be running.
pub enum Plane {
    /// Already in CPU memory (a plane the node passes through untouched).
    Host(Vec<u8>),
    /// A pinned buffer that a download from the GPU has been queued into.
    Pinned(PinnedHostSlice<u8>),
}

/// Waits for a point in the stream when dropped. It is the first field of [`GpuFrame`], so
/// the queued work is always finished before the buffers it uses are freed.
struct WaitOnDrop(CudaEvent);

impl Drop for WaitOnDrop {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}

/// A frame a GPU node has queued (upload, kernel and download) but not collected yet.
/// Made by [`PinnedPool::seal`], turned into a [`Frame`] by [`PinnedPool::finish`].
pub struct GpuFrame {
    // must stay the first field, see `WaitOnDrop`
    done: Option<WaitOnDrop>,
    width: u32,
    height: u32,
    y: Plane,
    u: Plane,
    v: Plane,
    a: Option<Plane>,
    /// GPU buffers the queued work reads and writes.
    keep: Vec<CudaSlice<u8>>,
    /// Pinned buffers the queued uploads read from.
    uploads: Vec<PinnedHostSlice<u8>>,
}

impl GpuFrame {
    /// Wraps a frame that needs no GPU work, so it can wait in the same slot as one that does.
    pub fn from_host(f: Frame) -> GpuFrame {
        return GpuFrame {
            done: None,
            width: f.width,
            height: f.height,
            y: Plane::Host(f.y),
            u: Plane::Host(f.u),
            v: Plane::Host(f.v),
            a: f.a.map(Plane::Host),
            keep: vec![],
            uploads: vec![],
        };
    }
}

/// Moves planes to and from the GPU through pinned (page-locked) host memory. That is the
/// only kind of host memory CUDA copies in the background, a copy to or from a plain `Vec`
/// blocks. Each GPU node owns one pool.
///
/// To queue a frame: `upload` each input plane, launch the kernel, `download` each output
/// plane, hand every GPU buffer to `keep`, then `seal`.
pub struct PinnedPool {
    stream: Arc<CudaStream>,
    /// Buffers that are free to reuse. Pinned memory is slow to allocate, so it is kept.
    free: Vec<PinnedHostSlice<u8>>,
    // collected for the frame being queued, moved into it by `seal`
    keep: Vec<CudaSlice<u8>>,
    uploads: Vec<PinnedHostSlice<u8>>,
}

impl PinnedPool {
    pub fn new(stream: Arc<CudaStream>) -> PinnedPool {
        return PinnedPool {
            stream,
            free: vec![],
            keep: vec![],
            uploads: vec![],
        };
    }

    fn take(&mut self, len: usize) -> PinnedHostSlice<u8> {
        if let Some(i) = self.free.iter().position(|b| b.len() == len) {
            return self.free.swap_remove(i);
        }
        // flags = 0 is plain pinned memory. cudarc's `alloc_pinned` asks for write-combined
        // memory, which is very slow for the CPU to read back.
        return unsafe {
            self.stream
                .context()
                .alloc_pinned_with_flags::<u8>(len, 0)
        }
        .unwrap();
    }

    /// Copies `plane` into a pinned buffer and queues its upload. Returns the GPU buffer
    /// straight away, the transfer happens in the background.
    pub fn upload(&mut self, plane: &[u8]) -> CudaSlice<u8> {
        let mut buf = self.take(plane.len());
        buf.as_mut_slice().unwrap().copy_from_slice(plane);
        let on_gpu = self.stream.clone_htod(&buf).unwrap();
        self.uploads.push(buf);
        return on_gpu;
    }

    /// Queues a download of `src` into a pinned buffer. Returns straight away.
    pub fn download(&mut self, src: &CudaSlice<u8>) -> Plane {
        let mut buf = self.take(src.len());
        self.stream.memcpy_dtoh(src, &mut buf).unwrap();
        return Plane::Pinned(buf);
    }

    /// Keeps a GPU buffer alive until the frame being queued has finished.
    pub fn keep(&mut self, buf: CudaSlice<u8>) {
        self.keep.push(buf);
    }

    /// Bundles everything queued since the last `seal` into one frame.
    pub fn seal(
        &mut self,
        width: u32,
        height: u32,
        y: Plane,
        u: Plane,
        v: Plane,
        a: Option<Plane>,
    ) -> GpuFrame {
        // marks the point in the stream where all of this frame's work is done
        let done = self.stream.record_event(None).unwrap();
        return GpuFrame {
            done: Some(WaitOnDrop(done)),
            width,
            height,
            y,
            u,
            v,
            a,
            keep: std::mem::take(&mut self.keep),
            uploads: std::mem::take(&mut self.uploads),
        };
    }

    /// Waits for whatever is left of the frame's GPU work (normally nothing) and copies
    /// the result out of pinned memory.
    pub fn finish(&mut self, g: GpuFrame) -> Frame {
        let GpuFrame {
            done,
            width,
            height,
            y,
            u,
            v,
            a,
            keep,
            uploads,
        } = g;
        // cudarc only tracks the event on a pinned buffer in multi-stream mode, so
        // wait on our own before reading any of them
        drop(done);
        drop(keep);
        self.free.extend(uploads);
        return Frame {
            width,
            height,
            y: self.collect(y),
            u: self.collect(u),
            v: self.collect(v),
            a: a.map(|a| self.collect(a)),
        };
    }

    fn collect(&mut self, plane: Plane) -> Vec<u8> {
        match plane {
            Plane::Host(v) => v,
            Plane::Pinned(buf) => {
                let v = buf.as_slice().unwrap().to_vec();
                self.free.push(buf);
                v
            }
        }
    }
}
pub trait BlankNode {
    fn next_frame(&mut self) -> Option<Frame>;
    fn reset(&mut self);
    fn node_hash(&mut self);
    fn get_hash(&self) -> HashOutput;
    fn fps(&mut self) -> Rational;
}
pub enum Node {
    Load(Load),
    Output(Output),
    Concat(Concat),
    ColorChange(ColorChange),
    Trim(Trim),
    Transform(Transform),
    Crop(Crop),
    Pancake(Pancake),
    Blur(Blur),
    SaturationChange(SaturationChange),
    Split(Split),
    FpsChange(FpsChange),
    Transition(Transition),
}

impl BlankNode for Node {
    fn next_frame(&mut self) -> Option<Frame> {
        return match self {
            Self::Load(l) => l.next_frame(),
            Self::Output(o) => o.next_frame(),
            Self::Concat(c) => c.next_frame(),
            Self::ColorChange(d) => d.next_frame(),
            Self::Trim(t) => t.next_frame(),
            Self::Transform(t) => t.next_frame(),
            Self::Crop(c) => c.next_frame(),
            Self::Pancake(p) => p.next_frame(),
            Self::Blur(b) => b.next_frame(),
            Self::SaturationChange(s) => s.next_frame(),
            Self::Split(s) => s.next_frame(),
            Self::FpsChange(f) => f.next_frame(),
            Self::Transition(t) => t.next_frame(),
        };
    }
    fn reset(&mut self) {
        match self {
            Self::Load(l) => l.reset(),
            Self::Output(o) => o.reset(),
            Self::Concat(c) => c.reset(),
            Self::ColorChange(d) => d.reset(),
            Self::Trim(t) => t.reset(),
            Self::Transform(t) => t.reset(),
            Self::Crop(c) => c.reset(),
            Self::Pancake(p) => p.reset(),
            Self::Blur(b) => b.reset(),
            Self::SaturationChange(s) => s.reset(),
            Self::Split(s) => s.reset(),
            Self::FpsChange(f) => f.reset(),
            Self::Transition(t) => t.reset(),
        };
    }
    fn node_hash(&mut self) {
        match self {
            Self::Load(l) => l.node_hash(),
            Self::Output(o) => o.node_hash(),
            Self::Concat(c) => c.node_hash(),
            Self::ColorChange(d) => d.node_hash(),
            Self::Trim(t) => t.node_hash(),
            Self::Transform(t) => t.node_hash(),
            Self::Crop(c) => c.node_hash(),
            Self::Pancake(p) => p.node_hash(),
            Self::Blur(b) => b.node_hash(),
            Self::SaturationChange(s) => s.node_hash(),
            Self::Split(s) => s.node_hash(),
            Self::FpsChange(f) => f.node_hash(),
            Self::Transition(t) => t.node_hash(),
        };
    }
    fn get_hash(&self) -> HashOutput {
        return match &self {
            &Self::Load(l) => l.get_hash(),
            Self::Output(o) => o.get_hash(),
            Self::Concat(c) => c.get_hash(),
            Self::ColorChange(d) => d.get_hash(),
            Self::Trim(t) => t.get_hash(),
            Self::Transform(t) => t.get_hash(),
            Self::Crop(c) => c.get_hash(),
            Self::Pancake(p) => p.get_hash(),
            Self::Blur(b) => b.get_hash(),
            Self::SaturationChange(s) => s.get_hash(),
            Self::Split(s) => s.get_hash(),
            Self::FpsChange(f) => f.get_hash(),
            Self::Transition(t) => t.get_hash(),
        };
    }
    fn fps(&mut self) -> Rational {
        return match self {
            Self::Load(l) => l.fps(),
            Self::Output(o) => o.fps(),
            Self::Concat(c) => c.fps(),
            Self::ColorChange(d) => d.fps(),
            Self::Trim(t) => t.fps(),
            Self::Transform(t) => t.fps(),
            Self::Crop(c) => c.fps(),
            Self::Pancake(p) => p.fps(),
            Self::Blur(b) => b.fps(),
            Self::SaturationChange(s) => s.fps(),
            Self::Split(s) => s.fps(),
            Self::FpsChange(f) => f.fps(),
            Self::Transition(t) => t.fps(),
        };
    }
}
