use crate::node::{BlankNode, Frame, HashOutput, Node};

use ffmpeg::{
    Dictionary, Packet, Rational, codec, encoder, format,
    format::Pixel,
    software::scaling::{context::Context as Scaler, flag::Flags},
    util::frame::video::Video,
};
use ffmpeg_next as ffmpeg;

pub struct Output {
    output_file_name: String,
    input_node: Box<Node>,
    hash: HashOutput,
}
impl Output {
    pub fn run(&mut self) {
        ffmpeg::init().unwrap();
        let fps = 30;

        let first = self.input_node.next_frame().expect("no frames");
        let h = first.len() as u32;
        let w = first[0].len() as u32;
        assert!(w % 2 == 0 && h % 2 == 0, "yuv420p needs even dimensions");

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

        let time_base = Rational(1, fps);
        enc.set_width(w);
        enc.set_height(h);
        enc.set_format(Pixel::YUV420P);
        enc.set_time_base(time_base);
        enc.set_frame_rate(Some(Rational(fps, 1)));
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

        let mut scaler =
            Scaler::get(Pixel::RGBA, w, h, Pixel::YUV420P, w, h, Flags::BILINEAR).unwrap();

        let mut next = Some(first);
        let mut i = 0;
        while let Some(frame) = next {
            // Fill an RGBA frame, respecting row stride padding
            let mut src = Video::new(Pixel::RGBA, w, h);
            let stride = src.stride(0);
            let dst = src.data_mut(0);
            for (y, row) in frame.iter().enumerate() {
                let line = &mut dst[y * stride..y * stride + w as usize * 4];
                for (x, px) in row.iter().enumerate() {
                    line[x * 4..x * 4 + 4].copy_from_slice(px);
                }
            }

            let mut yuv = Video::empty();
            scaler.run(&src, &mut yuv).unwrap();
            yuv.set_pts(Some(i as i64));

            enc.send_frame(&yuv).unwrap();
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
    pub fn new(output_file_name: &str, input_node: Node) -> Output {
        return Output {
            output_file_name: output_file_name.to_string(),
            input_node: Box::new(input_node),
            hash: 0,
        };
    }
}
impl BlankNode for Output {
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
        self.hash = (*self.input_node).get_hash();
    }
}
