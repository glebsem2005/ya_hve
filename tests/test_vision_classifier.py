"""Tests for cloud.vision.classifier — Vision API result parsing and safety nets."""

import json

from cloud.vision.classifier import _parse_result, VisionResult


class TestParseResult:
    def test_parse_result_has_machinery(self):
        """_parse_result extracts has_machinery from JSON."""
        raw = json.dumps(
            {
                "description": "Тяжёлая техника на вырубке",
                "has_human": False,
                "has_fire": False,
                "has_felling": True,
                "has_machinery": True,
                "is_threat": True,
            }
        )
        result = _parse_result(raw)
        assert isinstance(result, VisionResult)
        assert result.has_machinery is True

    def test_parse_result_has_machinery_default(self):
        """has_machinery defaults to False when missing from JSON."""
        raw = json.dumps(
            {
                "description": "Лес и тропинка",
                "has_human": False,
                "has_fire": False,
                "has_felling": False,
                "is_threat": False,
            }
        )
        result = _parse_result(raw)
        assert result.has_machinery is False

    def test_parse_result_human_felling_forces_threat(self):
        """Safety net: has_human + has_felling must force is_threat=True."""
        raw = json.dumps(
            {
                "description": "Женщина с бензопилой в лесу",
                "has_human": True,
                "has_fire": False,
                "has_felling": True,
                "is_threat": False,  # LLM said no threat — safety net should override
            }
        )
        result = _parse_result(raw)
        assert result.is_threat is True

    def test_parse_result_human_with_axe_desc_forces_threat(self):
        """Safety net: has_human + description mentions axe → is_threat=True."""
        raw = json.dumps(
            {
                "description": "Женщина, держащая в руках богато украшенный топор. Действие происходит в лесу.",
                "has_human": True,
                "has_fire": False,
                "has_felling": False,
                "is_threat": False,  # LLM said no threat
            }
        )
        result = _parse_result(raw)
        assert result.is_threat is True

    def test_parse_result_null_description(self):
        """description: null from API must not crash (AttributeError on .lower())."""
        raw = json.dumps(
            {
                "description": None,
                "has_human": True,
                "has_fire": False,
                "has_felling": True,
                "is_threat": False,
            }
        )
        result = _parse_result(raw)
        assert isinstance(result, VisionResult)
        assert result.description == ""
        # Safety net still fires: has_human + has_felling → is_threat
        assert result.is_threat is True

    def test_parse_result_malformed_json_returns_stub(self):
        """Malformed JSON returns conservative stub with is_threat=True."""
        result = _parse_result("this is not json at all {{{")
        assert isinstance(result, VisionResult)
        assert result.is_threat is True
        assert result.has_felling is True
        assert result.description != ""

    def test_parse_result_human_with_gun_forces_threat(self):
        """Safety net: has_human + 'ружьё' in description → is_threat=True."""
        raw = json.dumps(
            {
                "description": "Мужчина с ружьём идёт по лесной тропе",
                "has_human": True,
                "has_fire": False,
                "has_felling": False,
                "is_threat": False,
            }
        )
        result = _parse_result(raw)
        assert result.is_threat is True

    def test_parse_result_machinery_forces_threat(self):
        """Safety net: has_machinery=True must force is_threat=True even without humans."""
        raw = json.dumps(
            {
                "description": "Лесозаготовительные работы в хвойном лесу. Видна тяжелая техника.",
                "has_human": False,
                "has_fire": False,
                "has_felling": True,
                "has_machinery": True,
                "is_threat": False,  # LLM said no threat — safety net must override
            }
        )
        result = _parse_result(raw)
        assert result.is_threat is True
