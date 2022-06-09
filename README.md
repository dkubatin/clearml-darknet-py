<div align="center" style="text-align: center">

<p style="text-align: center">
  <img src="docs/clearml-darknet-logo.png" alt="ClearML Darknet">
</p>

**The library allows you to train neural networks on the [Darknet](https://github.com/AlexeyAB/darknet) framework in [ClearML](https://github.com/allegroai/clearml).**

![PyPI - Status](https://img.shields.io/pypi/status/clearml-darknet-py)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/clearml-darknet-py)
![PyPI](https://img.shields.io/pypi/v/clearml-darknet-py)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/clearml-darknet-py)
![GitHub](https://img.shields.io/github/license/dkubatin/clearml-darknet-py)

</div>

---

## Features

* **Graphs in ClearML** - Display graphs for any Darknet compatible networks.

    * For classifiers:
      * Accuracy (acc);
      * Average loss (avg_loss);
      * learning rate (lr).

    * For detectors:
      * Mean average precision 50% (mAP@0.50);
      * Precision;
      * Recall;
      * F1-score;
      * Average loss (avg_loss);
      * learning rate (lr).


* **Logs in ClearML** - Displays logs from Darknet in ClearML.


* **Saving weights in ClearML** - Ability to save weights in ClearML with flexible adjustment of parameters.

    Available options:
    * **Darknet**._is_save_weights_ - _Ability to disable saving weights (for experiments)_.
    * **Darknet**._save_weights_from_n_iterations_ - _Ability to save weights from the start of N iterations. For example, N=10000_.
    * **Darknet**._save_weights_every_n_iterations_ - _Ability to save weights every N iterations. For example, N=5000_.


* **Additional function of data splitting** - _Allows you to divide the data set into selections_.
  <details>
    <summary>📚 Click to see example</summary>
    
  ```python
  from clearml import Dataset
  from clearml_darknet.utils import split_dataset
  
  
  dataset_path = Dataset.get(dataset_name='dataset-example', dataset_project='Tests/darknet').get_local_copy()
  
  train, valid = split_dataset(
    dataset_path=dataset_path,
    ratio=0.7,
    shuffle=True
  )
  ```
  </details>

---

## Requirements

* Python (3.6)+
* <a href="https://github.com/allegroai/clearml" class="external-link" target="_blank">ClearML</a> (1.4.1)
* <a href="https://github.com/AlexeyAB/darknet" class="external-link" target="_blank">Darknet</a>


## Installation

_Before running, be sure to clone the Darknet repository and [compile](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make) for the appropriate operating system._

<div class="termy">

```console
$ pip install clearml-darknet-py
```

</div>


## Examples

* An example of a detector training script - [train_detector.py](https://github.com/dkubatin/clearml-darknet-py/tree/master/examples/train_detector.py)
* An example of a classificator training script - [train_classifier.py](https://github.com/dkubatin/clearml-darknet-py/tree/master/examples/train_classifier.py)
* Detector optimization script example - [optimize_detector.py](https://github.com/dkubatin/clearml-darknet-py/tree/master/examples/optimize_detector.py)

_For examples and use cases, check the [examples](https://github.com/dkubatin/clearml-darknet-py/tree/master/examples) folder._

---

## License
Clearml Darknet is MIT licensed as listed in the LICENSE file.
