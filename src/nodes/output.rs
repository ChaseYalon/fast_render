use crate::nodes::{BlankNode, Frame, HashOutput, Node};
use crate::project::ProjectConfig;
use std::hash::{DefaultHasher, Hash, Hasher};

use ffmpeg::{
    Dictionary, Packet, Rational, codec, encoder, format,
    format::Pixel,
    software::scaling::{context::Context as Scaler, flag::Flags},
    util::frame::video::Video,
};
use ffmpeg_next as ffmpeg;

pub struct Output {
    pub(crate) output_file_name: String,
    pub(crate) input_node: Box<Node>,
    hash: HashOutput,
    pub(crate) config: ProjectConfig,
}

/// Blends `value` over `background` by `alpha` (0 = all background, 255 = all value).
fn blend(value: u8, background: u8, alpha: u8) -> u8 {
    let (v, b, a) = (value as u32, background as u32, alpha as u32);
    return ((b * (255 - a) + v * a + 127) / 255) as u8;
}

/// Composites a frame with alpha onto black, so the encoder gets a fully opaque frame.
fn flatten(mut f: Frame) -> Frame {
    let Some(alpha) = f.a.take() else {
        return f;
    };
    // nothing is transparent, so there's nothing to blend
    if alpha.iter().all(|&a| a == 255) {
        return f;
    }
    // most pixels are fully opaque or fully transparent, so skip the blend math for those
    let apply = |value: &mut u8, background: u8, a: u8| match a {
        255 => {}
        0 => *value = background,
        _ => *value = blend(*value, background, a),
    };
    for (y, &a) in f.y.iter_mut().zip(&alpha) {
        apply(y, 16, a);
    }
    // each U/V sample covers two luma pixels, so use the average of their alphas
    let w = f.width as usize;
    let chroma_w = w / 2;
    for ((alpha_row, u_row), v_row) in alpha
        .chunks_exact(w)
        .zip(f.u.chunks_exact_mut(chroma_w))
        .zip(f.v.chunks_exact_mut(chroma_w))
    {
        for ((pair, u), v) in alpha_row.chunks_exact(2).zip(u_row).zip(v_row) {
            let a = ((pair[0] as u32 + pair[1] as u32 + 1) / 2) as u8;
            apply(u, 128, a);
            apply(v, 128, a);
        }
    }
    return f;
}

impl Output {
    pub fn run(&mut self) {
        ffmpeg::init().unwrap();
        let fps = self.config.fps;
        let input_fps = self.input_node.fps();
        assert_eq!(input_fps, fps, "the input's fps doesn't match the project's fps");

        let (w, h) = (self.config.width, self.config.height);
        assert!(w % 2 == 0 && h % 2 == 0, "the project size must be even");
        let first = self.input_node.next_frame().expect("no frames");

        let mut octx = format::output(&self.output_file_name).unwrap();
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let codec = encoder::find_by_name("libx264")
            .or_else(|| encoder::find(codec::Id::H264))
            .ok_or(ffmpeg::Error::EncoderNotFound)
            .unwrap();

        let mut ost = octx.add_stream(codec).unwrap();
        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .unwrap();

        let time_base = fps.invert();
        enc.set_width(w);
        enc.set_height(h);
        // Frames are 4:2:2 internally, but most players only decode 4:2:0 H.264
        enc.set_format(Pixel::YUV420P);
        enc.set_time_base(time_base);
        enc.set_frame_rate(Some(fps));
        if global_header {
            enc.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let mut opts = Dictionary::new();
        opts.set("preset", "medium");
        opts.set("crf", "18");

        let mut enc = enc.open_as_with(codec, opts).unwrap();
        ost.set_parameters(&enc);
        octx.write_header().unwrap();

        // The muxer may change the stream time base during write_header, so read it afterwards
        let stream_tb = octx.stream(0).unwrap().time_base();

        let cw = w as usize / 2;
        let mut to_420 =
            Scaler::get(Pixel::YUV422P, w, h, Pixel::YUV420P, w, h, Flags::BILINEAR).unwrap();

        let mut next = Some(first);
        let mut i = 0;
        while let Some(frame) = next {
            assert!(
                frame.width == w && frame.height == h,
                "frame {} is {}x{} but the project is {}x{}",
                i,
                frame.width,
                frame.height,
                w,
                h
            );
            let frame = flatten(frame);
            // Copy each plane into an ffmpeg frame, respecting row stride padding
            let mut yuv = Video::new(Pixel::YUV422P, w, h);
            for (plane, src, row_len) in [
                (0, &frame.y, w as usize),
                (1, &frame.u, cw),
                (2, &frame.v, cw),
            ] {
                let stride = yuv.stride(plane);
                let dst = yuv.data_mut(plane);
                for y in 0..h as usize {
                    dst[y * stride..y * stride + row_len]
                        .copy_from_slice(&src[y * row_len..(y + 1) * row_len]);
                }
            }
            let mut yuv420 = Video::empty();
            to_420.run(&yuv, &mut yuv420).unwrap();
            yuv420.set_pts(Some(i as i64));

            enc.send_frame(&yuv420).unwrap();
            self.drain(&mut enc, &mut octx, time_base, stream_tb);
            i += 1;
            next = self.input_node.next_frame();
        }

        enc.send_eof().unwrap();
        self.drain(&mut enc, &mut octx, time_base, stream_tb);
        octx.write_trailer().unwrap();
    }

    fn drain(
        &self,
        enc: &mut encoder::video::Encoder,
        octx: &mut format::context::Output,
        enc_tb: Rational,
        stream_tb: Rational,
    ) {
        let mut packet = Packet::empty();
        while enc.receive_packet(&mut packet).is_ok() {
            packet.set_stream(0);
            packet.rescale_ts(enc_tb, stream_tb);
            packet.write_interleaved(octx).unwrap();
        }
    }
    pub fn new(output_file_name: &str, input_node: Node, config: ProjectConfig) -> Output {
        return Output {
            output_file_name: output_file_name.to_string(),
            input_node: Box::new(input_node),
            hash: 0,
            config,
        };
    }
}
impl BlankNode for Output {
    fn fps(&mut self) -> Rational {
        return self.input_node.fps();
    }
    fn next_frame(&mut self) -> Option<Frame> {
        return None;
    }
    fn reset(&mut self) {
        self.input_node.reset();
    }
    fn get_hash(&self) -> HashOutput {
        return self.hash;
    }
    fn node_hash(&mut self) {
        self.input_node.node_hash();
        let mut h = DefaultHasher::new();
        self.input_node.get_hash().hash(&mut h);
        self.config.hash(&mut h);
        self.hash = h.finish();
    }
}
