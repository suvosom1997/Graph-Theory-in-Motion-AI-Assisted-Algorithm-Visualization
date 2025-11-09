import pytest
from voice.voice_utils import parse_voice_command


def test_parse_nodes_and_connectivity():
    txt = "Set nodes to 15 and edge density to 30 percent"
    res = parse_voice_command(txt)
    assert res.get('nodes') == 15
    assert abs(res.get('connectivity') - 0.3) < 1e-6


def test_parse_algorithm_and_start():
    txt = "Run BFS from node 3"
    res = parse_voice_command(txt)
    assert res.get('algorithm') == "Breadth-First Search (BFS)"
    assert res.get('start_node') == 3


def test_parse_word_numbers():
    txt = "Create twenty nodes"
    res = parse_voice_command(txt)
    assert res.get('nodes') == 20


def test_parse_no_input():
    res = parse_voice_command("")
    assert res == {}
