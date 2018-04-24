#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sandbox_fm
----------------------------------

Tests for `sandbox_fm` module.
"""

import logging

from click.testing import CliRunner

from sandbox_fm import cli

logger = logging.getLogger(__name__)


class TestSandbox_fm(object):
    def test_command_line_interface(self):
        runner = CliRunner()
        # run with test input
        result = runner.invoke(cli.run, [
            'models/zandmotor/zm_tide.mdu',
            '--max-iterations=1'
        ])
        assert result.exit_code == 0, (result)

    def test_help(self):
        runner = CliRunner()
        # run help
        help_result = runner.invoke(cli.run, ['--help'])
        assert help_result.exit_code == 0
        assert 'Show this message and exit.' in help_result.output

    @classmethod
    def teardown_class(cls):
        pass
