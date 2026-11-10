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
        let (w, h) = (first.width, first.height);
        assert!(w % 2 == 0, "yuv422p needs even width");

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
        // Frames are 4:2:2 internally, but most players only decode 4:2:0 H.264
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

        let cw = w as usize / 2;
        let mut to_420 =
            Scaler::get(Pixel::YUV422P, w, h, Pixel::YUV420P, w, h, Flags::BILINEAR).unwrap();

        let mut next = Some(first);
        let mut i = 0;
        while let Some(frame) = next {
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
