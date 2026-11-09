mod concat;
mod load;
mod node;
mod output;
use crate::{
    concat::Concat, load::Load, node::Node, output::Output,
};

fn main() {
    let v1 = Node::Load(Load::new("./media/vid1.mp4"));
    let v2 = Node::Load(Load::new("./media/vid2.mp4"));
    let c = Node::Concat(Concat::new(vec![v1, v2]));
    let mut o = Output::new("temp.mp4", c);
    o.run();
}
