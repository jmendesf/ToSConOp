# Connected operators based on the Tree of Shapes

## Presentation
This library allows the construction and modification of a Tree of Shapes (ToS) [1] based on the modification strategies introduced in [2].
Note that this implementation is only a proof-of-concept and is not updated anymore. 

The C++ implementation is the one that is currently being worked on and can be found here https://github.com/jmendesf/ToSConOpCpp.

This library allows the modification of the ToS such that (1) the resulting structure remains the ToS of the image it reconstructs and (2) the signs of the gradients between the regions of the image remains unchanged before and after modification.

## Prerequisites

The ToS is computed using the Higra [3] library, which can be downloaded here: https://github.com/higra/Higra. 

## Examples 

The notebook.ipynb file contains examples of tree of shapes construction and modification, as well as image reconstruction.

## References

[1] V. Caselles, B. Coll, and J. Morel, “Topographic maps and local contrast changes in natural images,” International Journal of Computer Vision, vol. 33, pp. 5–27, 1999.

[2] Julien Mendes Forte, Nicolas Passat, Akinobu Shimizu, Yukiko Kenmochi. Consistent connected operators based on trees of shapes. SIAM Journal on Imaging Sciences, 2025, 18 (4), pp.2547-2579. 

[3] Benjamin Perret, Giovanni Chierchia, Jean Cousty, Silvio Jamil F. Guimarães, Yukiko Kenmochi, et al.. Higra: Hierarchical Graph Analysis. SoftwareX, 2019, 10, pp.100335. 
