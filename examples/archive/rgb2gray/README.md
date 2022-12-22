# RGB to Grayscale
## Code logic:
- Essentially, read in mask only image (not composite) and figure out unique tuples that represent singular RGB values.
- A typical RBG value to produce a color value can be broken up into (R, G, B) insity values.
- Tuples are preferable over numpy arrays as they can be easily condensed into unique elemnts using a set and then
converted back into a list (they are hashable).
- After uniques are found, build a dictionary with values from 1 to number of unique RGB values in the images.
These will become the grayscale int values that we then plot.
The resulting array should automatically upscale from 8-bit integer if needed (>256 unique elements).
- **This only works for mask images, NOT overlays!**

![Comparison images](https://github.com/skalalab/rgb2gray/blob/master/sample_data/comparison.png?raw=true)