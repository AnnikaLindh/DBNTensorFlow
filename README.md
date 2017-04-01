# __DBNTensorFlow__ - _a Deep Belief Network implementation with TensorFlow_
Copyright (c) 2016 Annika Lindh
<br>Licensed under [GPLv3](http://www.gnu.org/licenses/), see LICENSE.txt

Implementations of a Restricted Boltzmann Machine and an unsupervised Deep Belief Network,
 including unsupervised fine-tuning of the Deep Belief Network.

### Publications
This code was used for two scientific publications. _(Yes, they have the same name. Lesson learned. :P)_
* __[Lindh, A. (2016). Investigating the Impact of Unsupervised Feature-Extraction from Multi-Wavelength Image Data for Photometric Classification of Stars, Galaxies and QSOs](http://arrow.dit.ie/scschcomdis/104/), *Dissertation, Dublin Institute of Technology.*__
<br>This is the full dissertation for my M.Sc. in Computing (Data Analytics) at Dublin Institute of Technology. It goes into details about the research itself as well as the project work.


* __[Lindh, A. (2016). Investigating the Impact of Unsupervised Feature-Extraction from Multi-Wavelength Image Data for Photometric Classification of Stars, Galaxies and QSOs.](http://arrow.dit.ie/scschcomcon/198/) *Proceedings of the 24th Irish Conference on Artificial Intelligence and Cognitive Science, Dublin, Ireland, p. 320-331.*__
<br>This is a more concise 12-page conference paper for the [24th Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2016)](http://aics2016.ucd.ie/). It focuses on the most important research findings.


### Requirements:

* Python 2.7
* tensorflow 0.8
* NumPy 1.11.0
* SciPy 0.13.3

The implementations have only been tested on Ubuntu but should
work on any OS that you can get TensorFlow running on.

---

 This was my first encounter with TensorFlow, so I started out from [Gabriele Angeletti's Deep-Learning-TensorFlow](https://github.com/blackecho/Deep-Learning-TensorFlow/) project which has implementations for a bunch of different Deep Learning techniques. I highly recommend a look at his GitHub repo! (If you were looking for a _supervised_ DBN, you will find that in his repo, while my implementation is of an _unsupervised_ DBN.)

The data used for this project was extracted from the multi-wavelength FITS image data from the [Sloan Digital Sky Survey](http://www.sdss.org/) database, Data Release 12.
Please see the publications for more details on the data selection and preprocessing.

Feel free to use this code for your own projects or for your personal learning, but I leave no guarantees of correctness or suitability in any way. Please see the LICENSE.txt document for the terms of this software. And I'd be happy if you give me credit in any projects and/or publications. :)

Bug reports and pull requests are welcome, though I may or may not have time to look at them.
<br>You can reach the author at code.annikalindh (at) gmail.com
