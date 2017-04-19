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

You probably want to create a virtual_ environment or an anaconda_ environment. Make sure you activate it, for example using the ``workon main`` command to activate your ``main`` environment (with virtualenvwrapper installed).

Once you are in your python environment of choice you can install the required python libraries:
* ``pip install -r requirements.txt``


Windows Install
----------------
- Download anaconda (3.5)
- Start the anaconda command window (from your start menu). If you do not have adminstrator rights you should start a command window using the following command (``%windir%\system32\cmd.exe "/K" "C:\Program Files\Anaconda3\Scripts\activate.bat" "C:\Program Files\Anaconda3"``)
- ``conda update conda``
- ``conda create --name main --file package-list-win.txt``
- ``activate main``  (You should now see a (main) at the start of your command line)
- ``conda install -c https://conda.binstar.org/menpo opencv3``
- Install the sandbox-fm software in develop mode ``pip install -e .`` (from the sandbox-fm directory)
- ``pip install tqdm``
- ``pip install -r requirements.txt``
- ``pip install cmocean``
- Make sure the dflowfm.dll is somewhere in your PATH definition

Running
-------

- ``sandbox-fm --help``
- ``sandbox-fm calibrate``  calibrate the sandbox by selecting a quad in the box, a quad in the model and a high and low point.
- ``sandbox-fm record``     record 10 frames, for testing
- ``sandbox-fm run``        run the sandbox program.
- ``sandbox-fm anomaly``    store the vertical anomaly for a plane in anomaly.npy
- ``sandbox-fm view``       view raw kinect images

Calibration
-----------
First run the command `sandbox-fm anomaly` with an empty sandbox to store the deviation from a plane.

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

Run
---

While running the simulation you can update the display using the following keys:

- 1 - Show bed level from camera
- 2 - Show water level in model
- 3 - Show bed level in model
- 4 - Show currents in model
- B – Set bed level to current camera bed level
- C – Currents on/off
- P – Photo
- F – Full screen view
- Q – Quit (windows only)
- R – Reset to original bathy

Features
--------

* TODO

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _macports: https://github.com/OpenKinect/libfreenect#fetch-build
.. _virtual: http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/
.. _anaconda: https://conda.io/docs/using/envs.html
