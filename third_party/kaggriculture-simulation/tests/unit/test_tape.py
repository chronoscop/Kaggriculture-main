import os

from kaggsim.tape import (action_to_line, line_to_action, read_tape,
                          tape_to_agent, validate_action, write_tape)


def test_empty_market_and_hand_slots_keep_their_position():
    a = {"farmer": ["WATER"], "hands": [[], ["NORTH"], []],
         "market": [[], ["SELL", "WHEAT", 3], [], ["HIRE"]]}
    line = action_to_line(a)
    assert line == "WATER\t;NORTH;\t;SELL WHEAT 3;;HIRE"
    back = line_to_action(line)
    assert back["hands"] == [[], ["NORTH"], []]
    assert back["market"] == [[], ["SELL", "WHEAT", 3], [], ["HIRE"]]


def test_non_list_entries_become_empty_segments():
    a = {"farmer": "PASS", "hands": [None, ["WEST"]],
         "market": ["junk", ["BUY_SEED", "WHEAT", 2]]}
    assert action_to_line(a) == "PASS\t;WEST\t;BUY_SEED WHEAT 2"


def test_counts_are_int_coerced_like_the_interpreter():
    a = {"market": [["SELL", "WHEAT", 2.9], ["SELL", "EGG", True]]}
    assert action_to_line(a).split("\t")[2] == "SELL WHEAT 2;SELL EGG 1"


def test_none_and_empty_actions():
    assert action_to_line(None) == "PASS\t\t"
    assert action_to_line({}) == "PASS\t\t"
    assert line_to_action("PASS\t\t") == {"farmer": ["PASS"], "hands": [],
                                          "market": []}


def test_unencodable_tokens_are_neutralised():
    a = {"farmer": ["PLANT", "WHE AT"], "market": [["SELL", "A;B", 1]]}
    assert action_to_line(a) == "PLANT ?\t\tSELL ? 1"


def test_validator():
    ok = {"farmer": ["PLANT", "WHEAT"], "hands": [[], ["WATER"]],
          "market": [[], ["SELL", "MILK", 2], ["HIRE"]]}
    assert validate_action(ok, n_hands=2) == []
    bad = {"farmer": ["FLY"], "hands": [["WATER"]],
           "market": [["SELL", "GOOSE", 1]] + [["HIRE"]] * 10}
    issues = " | ".join(validate_action(bad, n_hands=2))
    assert "unknown op 'FLY'" in issues
    assert "align positionally" in issues
    assert "11 market orders" in issues
    assert "ignored by the engine" in issues


def test_tape_round_trip_and_agent(tmp_path):
    p = str(tmp_path / "t.tape")
    write_tape(p, 5, [{"farmer": ["NORTH"], "market": [[], ["HIRE"]]},
                      "WATER\t\t"])
    seed, lines = read_tape(p)
    assert seed == 5 and lines == ["NORTH\t\t;HIRE", "WATER\t\t"]
    agent_path = tape_to_agent(p, str(tmp_path / "main.py"))
    ns = {}
    exec(open(agent_path, encoding="utf-8").read(), ns)
    assert ns["agent"]({"step": 0}) == {"farmer": ["NORTH"], "hands": [],
                                        "market": [[], ["HIRE"]]}
    assert ns["agent"]({"step": 5})["farmer"] == ["PASS"]
    assert os.path.getsize(agent_path) > 0


def test_tuples_are_encoded_like_lists():
    a = {"farmer": ("PLANT", "WHEAT"), "hands": [(), ("WATER",)],
         "market": (("SELL", "EGG", 2),)}
    assert action_to_line(a) == "PLANT WHEAT\t;WATER\tSELL EGG 2"


def test_control_characters_and_non_ascii_digits():
    # a form feed must not become a line break for anyone
    a = {"farmer": ["PLANT", "WHE" + chr(12) + "AT"],
         "market": [["SELL", "EGG", "2"], ["SELL", "EGG", chr(178)]]}
    line = action_to_line(a)
    assert chr(12) not in line
    assert line == "PLANT ?\t\tSELL EGG 2;SELL EGG ?"
    # superscript two is a digit to str.isdigit but not an integer token
    assert line_to_action("PASS\t\tSELL EGG " + chr(178))["market"] == [
        ["SELL", "EGG", chr(178)]]
    assert line_to_action("PASS\t\tSELL EGG +3")["market"] == [
        ["SELL", "EGG", 3]]


def test_read_tape_splits_on_newlines_only(tmp_path):
    p = tmp_path / "t.tape"
    p.write_bytes(b"SEED 1\r\nPASS\t\t\r\nWATER\t\t\n")
    assert read_tape(str(p)) == (1, ["PASS\t\t", "WATER\t\t"])


def test_validator_survives_non_string_ops():
    issues = validate_action({"farmer": [["PASS"]],
                              "market": [[{"x": 1}, "EGG", 1]]})
    assert any("unknown op" in i for i in issues)
    assert any("unknown order" in i for i in issues)
