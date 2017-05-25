"""Tests for qdyn_lb2diss CLI utility"""

import os
from os.path import join
import shutil
import filecmp

from click.testing import CliRunner

from QDYN._lb2diss import main
# built-in fixtures: tmpdir, request


def test_qdyn_lb2diss(tmpdir, request):
    """Test invocation of qdyn_lb2diss"""
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    runner = CliRunner()

    rf_in = os.path.join(test_dir, 'rf')

    rf = str(tmpdir.join("rf1"))
    config = join(rf, 'config')
    shutil.copytree(rf_in, rf)
    result = runner.invoke(
        main, ['--overwrite', config])
    assert result.exit_code == 1
    assert "ERROR" in result.output

    rf = str(tmpdir.join("rf2"))
    config = join(rf, 'config')
    shutil.copytree(rf_in, rf)
    result = runner.invoke(
        main, ['--label=dissipative', '--overwrite', config])
    assert result.exit_code == 0
    assert result.output == ''
    assert filecmp.cmp(join(rf_in, 'config'), config + "~")
    assert filecmp.cmp(
        join(test_dir, 'expected', 'D_dissipative.dat'),
        join(rf, 'D_dissipative.dat'))
    assert filecmp.cmp(join(test_dir, 'expected', 'config'), config)
    assert os.path.isfile(join(rf, 'A_decay.dat'))  # wasn't deleted

    rf = str(tmpdir.join("rf3"))
    config = join(rf, 'config')
    shutil.copytree(rf_in, rf)
    result = runner.invoke(
        main, ['--label=dissipative', '--bak=.bak', '--overwrite', config])
    assert result.exit_code == 0
    assert result.output == ''
    assert filecmp.cmp(join(rf_in, 'config'), config + ".bak")

    rf = str(tmpdir.join("rf4"))
    config = join(rf, 'config')
    shutil.copytree(rf_in, rf)
    result = runner.invoke(
        main, ['--label=dissipative', config])
    assert result.exit_code == 0
    assert filecmp.cmp(join(rf_in, 'config'), config)  # unchanged
    with open(join(test_dir, 'expected', 'config')) as in_fh:
        assert result.output.strip() == in_fh.read().strip()

    rf = str(tmpdir.join("rf5"))
    config = join(rf, 'config')
    shutil.copytree(rf_in, rf)
    result = runner.invoke(
        main,
        ['--label=dissipative', '--no-bak', '--delete', '--overwrite', config])
    assert result.exit_code == 0
    assert result.output == ''
    assert filecmp.cmp(join(test_dir, 'expected', 'config'), config)
    assert not os.path.isfile(config + "~")
    assert not os.path.isfile(join(rf, 'A_deph.dat'))
    assert not os.path.isfile(join(rf, 'A_decay.dat'))

    rf = str(tmpdir.join("rf6"))
    config = join(rf, 'config')
    shutil.copytree(rf_in, rf)
    result = runner.invoke(
        main,
        ['--label=dissipative', '--overwrite', '--dissfile', 'diss.dat',
         config])
    assert result.exit_code == 0
    assert filecmp.cmp(
        join(test_dir, 'expected', 'D_dissipative.dat'),
        join(rf, 'diss.dat'))
