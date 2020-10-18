import pytest
import importlib


def test_intro():
    numpy_spec = importlib.util.find_spec("numpy")
    torch_spec = importlib.util.find_spec("torch")
    torchgeo_spec = importlib.util.find_spec("torch_geometric")
    assert numpy_spec is not None, "Numpy not found"
    assert torch_spec is not None, "Pytorch not found"
    assert torchgeo_spec is not None, "Pytorch Geometric not found"


def test_KG():
    numpy_spec = importlib.util.find_spec("numpy")
    torch_spec = importlib.util.find_spec("torch")
    torchgeo_spec = importlib.util.find_spec("torch_geometric")
    sklearn_spec = importlib.util.find_spec("sklearn")
    matplotlib_spec = importlib.util.find_spec("matplotlib")
    assert numpy_spec is not None, "Numpy not found"
    assert torch_spec is not None, "Pytorch not found"
    assert torchgeo_spec is not None, "Pytorch Geometric not found"
    assert sklearn_spec is not None, "Sklearn not found"
    assert matplotlib_spec is not None, "Matplotlib Geometric not found"


def test_graph_classification():
    numpy_spec = importlib.util.find_spec("numpy")
    torch_spec = importlib.util.find_spec("torch")
    torchgeo_spec = importlib.util.find_spec("torch_geometric")
    sklearn_spec = importlib.util.find_spec("sklearn")
    matplotlib_spec = importlib.util.find_spec("matplotlib")
    networkx_spec = importlib.util.find_spec("networkx")
    tqdm_spec = importlib.util.find_spec("tqdm")
    dgl_spec = importlib.util.find_spec("dgl")
    assert numpy_spec is not None, "Numpy not found"
    assert torch_spec is not None, "Pytorch not found"
    assert torchgeo_spec is not None, "Pytorch Geometric not found"
    assert sklearn_spec is not None, "Sklearn not found"
    assert matplotlib_spec is not None, "Matplotlib Geometric not found"
    assert networkx_spec is not None, "NetworkX not found"
    assert tqdm_spec is not None, "TQDM is not found"
    assert dgl_spec is not None, "DGL is not found"


def test_link_prediction():
    numpy_spec = importlib.util.find_spec("numpy")
    torch_spec = importlib.util.find_spec("torch")
    torchgeo_spec = importlib.util.find_spec("torch_geometric")
    sklearn_spec = importlib.util.find_spec("sklearn")
    matplotlib_spec = importlib.util.find_spec("matplotlib")
    opentsne_spec = importlib.util.find_spec("openTSNE")
    assert numpy_spec is not None, "Numpy not found"
    assert torch_spec is not None, "Pytorch not found"
    assert torchgeo_spec is not None, "Pytorch Geometric not found"
    assert sklearn_spec is not None, "Sklearn not found"
    assert matplotlib_spec is not None, "Matplotlib Geometric not found"
    assert opentsne_spec is not None, "Open TSNE not found"


def test_graph_attention():
    numpy_spec = importlib.util.find_spec("numpy")
    torch_spec = importlib.util.find_spec("torch")
    torchgeo_spec = importlib.util.find_spec("torch_geometric")
    sklearn_spec = importlib.util.find_spec("sklearn")
    matplotlib_spec = importlib.util.find_spec("matplotlib")
    assert numpy_spec is not None, "Numpy not found"
    assert torch_spec is not None, "Pytorch not found"
    assert torchgeo_spec is not None, "Pytorch Geometric not found"
    assert sklearn_spec is not None, "Sklearn not found"
    assert matplotlib_spec is not None, "Matplotlib Geometric not found"
