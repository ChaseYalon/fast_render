use crate::node::{BlankNode, Frame, HashOutput};
use ffmpeg::{
    Packet,
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

        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters()).unwrap();
        let mut decoder = ctx.decoder().video().unwrap();
        let (w, h) = (decoder.width(), decoder.height());

        let scaler =
            Scaler::get(decoder.format(), w, h, Pixel::RGBA, w, h, Flags::BILINEAR).unwrap();

        return Decoding { ictx, decoder, scaler, idx, w, h, eof: false };
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
    fn next_frame(&mut self) -> Option<Frame> {
        if self.state.is_none() {
            self.state = Some(self.open());
        }
        let s = self.state.as_mut().unwrap();
        let mut decoded = Video::empty();
        let mut packet = Packet::empty();
        loop {
            if s.decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgba = Video::empty();
                s.scaler.run(&decoded, &mut rgba).unwrap();

                // rows may be padded, so index by stride rather than w * 4
                let stride = rgba.stride(0);
                let data = rgba.data(0);
                return Some(
                    (0..s.h as usize)
                        .map(|y| {
                            let row = &data[y * stride..y * stride + s.w as usize * 4];
                            row.chunks_exact(4)
                                .map(|p| [p[0], p[1], p[2], p[3]])
                                .collect()
                        })
                        .collect(),
                );
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
