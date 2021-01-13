# Turbulent Fluid Flows with Generative Deep Learning
Multi-fidelity Generative Deep Learning Turbulent Flows [[FoDS](http://aimsciences.org//article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097)][[ArXiv](https://arxiv.org/abs/2006.04731)]

[Nicholas Geneva](http://nicholasgeneva.com/), [Nicholas Zabaras](https://cics.nd.edu)

---
[![Documentation Status](https://readthedocs.org/projects/deep-turbulence/badge/?version=latest)](https://deep-turbulence.readthedocs.io/en/latest/?badge=latest) [![dataset](https://zenodo.org/badge/DOI/10.5281/zenodo.4311698.svg)](https://doi.org/10.5281/zenodo.4311698) [![liscense](https://img.shields.io/github/license/zabaras/deep-turbulence)](https://github.com/zabaras/deep-turbulence/blob/master/LICENSE)

A novel multi-fidelity deep generative model is introduced for the surrogate modeling of high-fidelity turbulent flow fields given the solution of a computationally inexpensive but inaccurate low-fidelity solver.

- [Getting Started](https://deep-turbulence.readthedocs.io/en/latest/start.html)
- [Documentation](https://deep-turbulence.readthedocs.io/en/latest/index.html)
- [Data Repository](https://doi.org/10.5281/zenodo.4298896)

### Core Dependencies
* Python 3.6.5
* [PyTorch](https://pytorch.org/) 1.6.0
* [Matplotlib](https://matplotlib.org/) 3.1.1
* [SciPy](https://www.scipy.org/) 1.5.2
* [Dataclasses](https://docs.python.org/3/library/dataclasses.html) 0.7.0

See requirements.txt for full dependency list.

## Citation
Find this useful or like this work? Cite us with:
```bibtex
    @article{geneva2020multi,
        title = "Multi-fidelity generative deep learning turbulent flows",
            author = "Nicholas Geneva and  Nicholas Zabaras",
            journal = "Foundations of Data Science",
            volume = "2",
            pages = "391",
            year = "2020",
            issn = "A0000-0002",
            doi = "10.3934/fods.2020019",
            url = "http://aimsciences.org/article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097"
    }
```