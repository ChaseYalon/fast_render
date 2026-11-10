use crate::node::{BlankNode, Frame, HashOutput};
use ffmpeg::{
    Packet, Rational,
    format::{Pixel, input},
    media::Type,
    software::scaling::{context::Context as Scaler, flag::Flags},
    util::frame::video::Video,
};
use ffmpeg_next as ffmpeg;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::SystemTime;
struct Decoding {
    ictx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    scaler: Scaler,
    idx: usize,
    w: u32,
    h: u32,
    fps: Rational,
    eof: bool,
}
pub struct Load {
    pub file_path: String,
    pub hash: HashOutput,
    state: Option<Decoding>,
}
impl Load {
    fn open(&self) -> Decoding {
        ffmpeg::init().unwrap();

        let mut ictx = input(&self.file_path).unwrap();
        let stream = ictx.streams().best(Type::Video).expect("no video stream");
        let idx = stream.index();
        let fps = stream.avg_frame_rate();
        assert!(
            fps.numerator() > 0 && fps.denominator() > 0,
            "could not read the frame rate of {}",
            self.file_path
        );

        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters()).unwrap();
        let mut decoder = ctx.decoder().video().unwrap();
        let (w, h) = (decoder.width(), decoder.height());

        let scaler = Scaler::get(
            decoder.format(),
            w,
            h,
            Pixel::YUV422P,
            w,
            h,
            Flags::BILINEAR,
        )
        .unwrap();

        return Decoding {
            ictx,
            decoder,
            scaler,
            idx,
            w,
            h,
            fps,
            eof: false,
        };
    }
    pub fn new(path: &str) -> Load {
        return Load {
            file_path: path.to_string(),
            hash: 0,
            state: None,
        };
    }
}
impl BlankNode for Load {
    fn fps(&mut self) -> Rational {
        if self.state.is_none() {
            self.state = Some(self.open());
        }
        return self.state.as_ref().unwrap().fps;
    }
    fn next_frame(&mut self) -> Option<Frame> {
        if self.state.is_none() {
            self.state = Some(self.open());
        }
        let s = self.state.as_mut().unwrap();
        let mut decoded = Video::empty();
        let mut packet = Packet::empty();
        loop {
            if s.decoder.receive_frame(&mut decoded).is_ok() {
                let mut yuv = Video::empty();
                s.scaler.run(&decoded, &mut yuv).unwrap();

                // rows may be padded, so copy row by row using each plane's stride
                let cw = (s.w as usize + 1) / 2;
                let copy_plane = |plane: usize, row_len: usize| -> Vec<u8> {
                    let stride = yuv.stride(plane);
                    let data = yuv.data(plane);
                    let mut out = Vec::with_capacity(row_len * s.h as usize);
                    for y in 0..s.h as usize {
                        out.extend_from_slice(&data[y * stride..y * stride + row_len]);
                    }
                    out
                };
                return Some(Frame {
                    width: s.w,
                    height: s.h,
                    y: copy_plane(0, s.w as usize),
                    u: copy_plane(1, cw),
                    v: copy_plane(2, cw),
                });
            }
            if s.eof {
                return None;
            }
            match packet.read(&mut s.ictx) {
                Ok(()) => {
                    if packet.stream() == s.idx {
                        s.decoder.send_packet(&packet).unwrap();
                    }
                }
                Err(_) => {
                    s.decoder.send_eof().unwrap();
                    s.eof = true;
                }
            }
        }
    }
    fn reset(&mut self) {
        self.state = None;
    }
    fn node_hash(&mut self) {
        let mut h = DefaultHasher::new();
        self.file_path.hash(&mut h);
        let metadata = fs::metadata(self.file_path.clone()).unwrap();

        let time = metadata.modified().unwrap();
        let edited_at = time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        edited_at.hash(&mut h);
        self.hash = h.finish();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
}
