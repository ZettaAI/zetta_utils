from zetta_utils.layer import Layer


def test_with_procs(mocker):
    backend = mocker.MagicMock()
    read_proc = mocker.MagicMock()
    write_proc = mocker.MagicMock()
    index_proc = mocker.MagicMock()

    layer1 = Layer(backend=backend)
    layer2 = layer1.with_procs(
        read_procs=[read_proc], write_procs=[write_proc], index_procs=[index_proc]
    )
    assert layer2.read_procs == (read_proc,)
    assert layer2.write_procs == (write_proc,)
    assert layer2.index_procs == (index_proc,)
