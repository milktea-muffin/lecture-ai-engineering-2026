"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai import DeferredToolRequests
from pydantic_ai.models.test import TestModel
import pytest

from app import tools
from app.agent import agent, resolve_approvals

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


@pytest.mark.parametrize("answer", ["y", "n", "", "yes", None])
def test_learning_log_requires_approval(tmp_path, monkeypatch, answer):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    model = TestModel(call_tools=["save_learning_log"], seed=1)
    result = agent.run_sync("学習内容を保存して", model=model)
    assert isinstance(result.output, DeferredToolRequests)
    assert not log_dir.exists()
    call = result.output.approvals[0]
    assert call.tool_name == "save_learning_log"

    def respond(prompt):
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", respond)
    final = resolve_approvals(agent, result, model)
    assert isinstance(final.output, str)
    if answer == "y":
        files = list(log_dir.glob("*.md"))
        assert len(files) == 1
        args = call.args_as_dict()
        assert files[0].read_text(encoding="utf-8") == (
            f"# 第{args['lecture']}回 学習記録\n\n{args['content']}\n"
        )
    else:
        assert not log_dir.exists()
