# Script syntax

A script is a list of lines ending in `;`. Every node is written `NAME(param=value, ...)`.
Parameters are always named, in any order, and all of them are required.

```
PROJECT(width=1920, height=1080, fps=30);

l1 = LOAD(file="media/vid1.mp4");
t1 = TRIM(input=l1, start=0, end=10);
b1 = BLUR(input=t1, sigma=4);

OUTPUT(file="out.mp4", input=b1);
```

## The basics

- **Node names** are all caps (`LOAD`, `FPS_CHANGE`). **Variable names** have a lowercase letter (`l1`, `fps_c1`).
- `name = NODE(...);` saves a node in a variable. Numbers can be saved too: `half = PROJECT.width / 2;`.
- A variable has to be assigned once, and can be used anywhere in the file. Using it twice makes two independent copies of it.
- Numbers can be written as `5`, `2.5`, `PROJECT.width`, or a mix with `+ - * /` and `( )`. Negatives work: `-5`.
- Strings are `"in quotes"`, with no escapes, so a Windows path like `"C:\media\a.mp4"` works as is.
- Times (`start`, `end`, `at`, `seconds`) are in seconds. Sizes and positions are in pixels.
- There are no comments.

## The two required lines

### `PROJECT(width, height, fps)`
The canvas every frame has to match. All three are plain integers (no expressions).
Width and height must be even. Its values can be used elsewhere as `PROJECT.width`, `PROJECT.height` and `PROJECT.fps`.

### `OUTPUT(file, input)`
Renders `input` to `file`. Exactly one per script. `input`'s fps has to equal `PROJECT`'s fps
(use `FPS_CHANGE` to get there).

## Nodes

| Node | Parameters | What it does |
|---|---|---|
| `LOAD` | `file` | Reads a video file. |
| `TRIM` | `input`, `start`, `end` | Keeps `start` up to (not including) `end`. Whole seconds, `start` < `end`. |
| `CROP` | `input`, `left_x`, `left_y`, `right_x`, `right_y` | Makes everything outside the window transparent, the frame keeps its size. Top left is included, bottom right is not. |
| `CONCAT` | `input1`, `input2`, ... | Plays the inputs one after the other. Numbered from 1 with no gaps, all the same fps. |
| `SPLIT` | `input`, `at` | Cuts a node in two at `at` seconds. Used through `.first` / `.second`, see below. |
| `FPS_CHANGE` | `input`, `new_fps` | Repeats or drops frames to reach `new_fps` (a whole number). The clip keeps its length. |
| `COLOR_CHANGE` | `input`, `strength` | Colour change of the whole frame by `strength`. |
| `SATURATION_CHANGE` | `input`, `factor` | `0` is grayscale, `1` is unchanged, `2` doubles the colour. Can't be negative. |
| `BLUR` | `input`, `sigma` | Gaussian blur. `sigma` has to be above 0. |
| `TRANSFORM` | `input`, `trans_x`, `trans_y`, `rot_x`, `rot_y`, `deg_of_rot`, `scale_x`, `scale_y`, `scale_factor` | Moves by `trans_x/y`, rotates `deg_of_rot` degrees around `(rot_x, rot_y)`, scales by `scale_factor` around `(scale_x, scale_y)`. `scale_factor` can't be 0. |
| `PANCAKE` | `bottom`, `top` | Layers `top` over `bottom` using their transparency. Same size and fps. |
| `TRANSITION` | `a`, `b`, `kind`, `seconds` | Plays `a`, then `b`, blending the last `seconds` of `a` into the start of `b`. The result is `seconds` shorter than the two back to back. Same size and fps. |

### `TRANSITION` kinds
`kind` is a name, not a string: `CROSSFADE`, `DIP_TO_BLACK`, `WIPE_LEFT_TO_RIGHT`, `WIPE_RIGHT_TO_LEFT`.

```
TRANSITION(a=l1, b=l2, kind=CROSSFADE, seconds=1)
```

### `SPLIT`
`SPLIT` makes two nodes, so it can't be used on its own. Take a half of it with `.first` (before the cut) or `.second` (after it):

```
s = SPLIT(input=l1, at=5);
joined = CONCAT(input1=s.second, input2=s.first);
```

Or without a variable: `SPLIT(input=l1, at=5).first`.

## Nesting

A node can go straight into a parameter instead of through a variable:

```
OUTPUT(file="out.mp4", input=BLUR(input=TRIM(input=LOAD(file="media/vid1.mp4"), start=0, end=5), sigma=2));
```

## Mistakes

A bad script stops with a panic. The messages for a missing `PROJECT` / `OUTPUT` line or a bad `PROJECT`
value say what is wrong. Most other mistakes (unknown node name, missing parameter, a number where a
node is needed) are a plain `unreachable` or `no entry found for key`.
