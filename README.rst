===============================
Sandbox Delft3D Flexible Mesh
===============================


.. image:: https://img.shields.io/pypi/v/sandbox_fm.svg
        :target: https://pypi.python.org/pypi/sandbox_fm

.. image:: https://img.shields.io/travis/openearth/sandbox_fm.svg
        :target: https://travis-ci.org/openearth/sandbox_fm

.. image:: https://readthedocs.org/projects/sandbox-fm/badge/?version=latest
        :target: https://sandbox-fm.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/openearth/sandbox_fm/shield.svg
     :target: https://pyup.io/repos/github/openearth/sandbox_fm/
     :alt: Updates


Sandbox combined with a Delft3D Flexbile Mesh simulation

* Free software: GNU General Public License v3
* Documentation: https://sandbox-fm.readthedocs.io.

Install
-------
After you have created a sandbox you can hook up the software. Before you can use the software you have to install some prerequisites.
These include the following packages:
* python (tested with 2.7 and 3.5)

You preferably want to install these from the package manager (tested with ubuntu and osx + macports):
* libfreenect (available by default in linux and there are instructions for use with .. _macports: macports, make sure you build the python bindings, it is also available in homebrew)
* opencv (available in linux and osx (macports/homebrew)

Several python libraries:
* pip install -r requirements.txt


Windows Install
----------------
- Download anaconda (3.5)
- Start the anaconda command window (from your start menu)
- Update anaconda: `conda update conda`
- Create an environment: `conda create --name main python=3`
- Activate the new environment `activate main`
- Install development dependencies `pip install -r requirements_dev.txt`
- Install dependencies `pip install -r requirements.txt`
- Install the sandbox-fm software in develop mode `pip install -e .` (from the sandbox-fm directory)
- Make sure the dflowfm.dll is somewhere in your PATH

Running
-------

sandbox-fm --help
sandbox-fm calibrate  calibrate the sandbox by selecting a quad in the box, a quad in the model and a high and low point.
sandbox-fm record     record 10 frames, for testing
sandbox-fm run        run the sandbox program.
sandbox-fm view       view raw kinect images


Calibration
-----------

Calibration transforms between the different quads.

- Photo from kinect (video)
- Depth from kinect (img)
- Beamer (box, default 640x480)
- Model extract (model)

In the top left window select the extent of the beamer.
In the top right window select a corresponding extent in the model.
In the bottom left model select a low and a high point.
Press [ENTER].
Done.

Features
--------

* TODO

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _macports: https://github.com/OpenKinect/libfreenect#fetch-build
