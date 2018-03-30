#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sandbox_fm
----------------------------------

Tests for `sandbox_fm` module.
"""

import pytest

import logging

from contextlib import contextmanager
from click.testing import CliRunner

from sandbox_fm import sandbox_fm
from sandbox_fm import cli

logger = logging.getLogger(__name__)


def log_ex(ex):
    try:
        raise ex
    except:
        logger.exception("exception raised")



class TestSandbox_fm(object):

    @classmethod
    def setup_class(cls):
        pass

    def test_something(self):
        pass

    def test_command_line_interface(self):
        runner = CliRunner()

        # run with test input
        result = runner.invoke(cli.run, [
            'tests/zandmotor/zm_tide.mdu',
            '--max-iterations=1'
        ])
        logger.info(result.output)
        assert result.exit_code == 0, (result)
        assert 'sandbox' in result.output.lower()


    def test_help(self):
        runner = CliRunner()
        # run help
        help_result = runner.invoke(cli.run, ['--help'])
        assert help_result.exit_code == 0
        assert 'Show this message and exit.' in help_result.output

    @classmethod
    def teardown_class(cls):
        pass
