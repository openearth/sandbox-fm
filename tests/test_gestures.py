#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_gestures
----------------------------------

Tests for `gestures` module.
"""

import logging

import numpy as np

from sandbox_fm.gestures import (
    recognize_gestures
)

logger = logging.getLogger(__name__)


class TestGestures(object):
    @classmethod
    def setup_class(cls):
        pass

    def test_no_hand(self):
        height = np.zeros(shape=(100, 100))
        gestures = recognize_gestures(height)
        assert len(gestures) == 0, 'should have no hand'

    @classmethod
    def teardown_class(cls):
        pass
