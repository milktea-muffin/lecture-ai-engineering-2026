"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_make_quiz_uses_a_term_from_the_requested_lecture():
    quiz = json.loads(tools.make_quiz(2))
    assert set(quiz) == {"question", "answer"}
    assert any(
        quiz == {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"]}
        for entry in tools.load_glossary()
        if entry["lecture"] == 2
    )


def test_make_quiz_reports_missing_lecture():
    assert "用語集にありません" in tools.make_quiz(99)


def test_learning_logs_are_saved_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    tools.save_learning_log(3, "ツールの登録方法を学んだ。")
    tools.save_learning_log(3, "保存には承認が必要。")
    files = list(tools.LEARNING_LOG_DIR.glob("*.md"))
    assert len(files) == 2
    assert {p.read_text(encoding="utf-8") for p in files} == {
        "# 第3回 学習記録\n\nツールの登録方法を学んだ。\n",
        "# 第3回 学習記録\n\n保存には承認が必要。\n",
    }


def test_learning_log_rejects_invalid_input(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    assert "保存できません" in tools.save_learning_log(0, "学習内容")
    assert "保存できません" in tools.save_learning_log(3, "  ")
    assert not tools.LEARNING_LOG_DIR.exists()


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"
