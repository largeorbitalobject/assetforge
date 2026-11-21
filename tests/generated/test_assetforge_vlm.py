import json
import os
import sys
import types
import urllib.request
from unittest.mock import patch, MagicMock

import pytest

# Ensure the module can be imported even if the package structure isn't fully set up
# We'll mock the necessary parts if needed, but first let's try importing directly.
# The module uses `from __future__ import annotations` and standard libs.

# We need to be careful: the module executes code at import time (ENDPOINT, MODEL).
# It does not make network calls at import time.

import assetforge.vlm as vlm


class TestB64:
    def test_b64_encodes_file(self, tmp_path):
        # Create a temporary file with known content
        test_file = tmp_path / "test.png"
        content = b"fake image data"
        test_file.write_bytes(content)
        
        result = vlm._b64(str(test_file))
        
        import base64
        expected = base64.b64encode(content).decode()
        assert result == expected


class TestJsonBlock:
    def test_json_block_extracts_list(self):
        text = 'Here is the result: ["tree", "rock"]'
        result = vlm._json_block(text)
        assert result == ["tree", "rock"]

    def test_json_block_extracts_dict(self):
        text = 'Result: {"base": "ground", "joints": []}'
        result = vlm._json_block(text)
        assert result == {"base": "ground", "joints": []}

    def test_json_block_returns_none_if_no_json(self):
        text = "No JSON here, just text."
        result = vlm._json_block(text)
        assert result is None

    def test_json_block_extracts_nested_json(self):
        text = 'Some text {"a": [1, 2, 3]} more text'
        result = vlm._json_block(text)
        assert result == {"a": [1, 2, 3]}

    def test_json_block_extracts_multiline_json(self):
        text = '''
        {
            "key": "value"
        }
        '''
        result = vlm._json_block(text)
        assert result == {"key": "value"}


class TestAvailable:
    def test_available_returns_true_on_success(self):
        with patch.object(urllib.request, 'urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            mock_urlopen.return_value.read.return_value = b'[]'
            # The function calls urlopen with a modified endpoint
            result = vlm.available()
            assert result is True
            # Verify it called urlopen with the /models endpoint
            args, kwargs = mock_urlopen.call_args
            assert "/models" in args[0]

    def test_available_returns_false_on_exception(self):
        with patch.object(urllib.request, 'urlopen') as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")
            result = vlm.available()
            assert result is False


class TestChat:
    def test_chat_constructs_correct_payload(self):
        with patch.object(urllib.request, 'urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "Hello"}}]
            }).encode()
            mock_urlopen.return_value = mock_response

            result = vlm.chat("Test prompt")
            
            assert result == "Hello"
            
            # Verify the request payload
            args, kwargs = mock_urlopen.call_args
            req = args[0]
            payload = json.loads(req.data.decode())
            
            assert payload["model"] == vlm.MODEL
            assert payload["temperature"] == 0.0
            assert payload["max_tokens"] == 512
            assert payload["messages"][0]["role"] == "user"
            assert payload["messages"][0]["content"][0]["type"] == "text"
            assert payload["messages"][0]["content"][0]["text"] == "Test prompt"

    def test_chat_includes_images(self, tmp_path):
        # Create a dummy image file
        img_file = tmp_path / "img.png"
        img_file.write_bytes(b"fake image")
        
        with patch.object(urllib.request, 'urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "Image processed"}}]
            }).encode()
            mock_urlopen.return_value = mock_response

            result = vlm.chat("Describe image", images=[str(img_file)])
            
            assert result == "Image processed"
            
            args, kwargs = mock_urlopen.call_args
            req = args[0]
            payload = json.loads(req.data.decode())
            
            content = payload["messages"][0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_chat_custom_model_and_params(self):
        with patch.object(urllib.request, 'urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps({
                "choices": [{"message": {"content": "Custom"}}]
            }).encode()
            mock_urlopen.return_value = mock_response

            result = vlm.chat("Prompt", temperature=0.5, max_tokens=100, model="custom-model")
            
            assert result == "Custom"
            
            args, kwargs = mock_urlopen.call_args
            req = args[0]
            payload = json.loads(req.data.decode())
            
            assert payload["model"] == "custom-model"
            assert payload["temperature"] == 0.5
            assert payload["max_tokens"] == 100


class TestNameHeroes:
    def test_name_heroes_returns_list(self):
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = '["tree", "rock", "bush"]'
            
            result = vlm.name_heroes("dummy.png")
            
            assert result == ["tree", "rock", "bush"]
            mock_chat.assert_called_once()
            args, kwargs = mock_chat.call_args
            assert "dummy.png" in kwargs.get("images", [])

    def test_name_heroes_returns_empty_list_on_invalid_json(self):
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = "Not a JSON array"
            
            result = vlm.name_heroes("dummy.png")
            
            assert result == []

    def test_name_heroes_returns_empty_list_on_dict(self):
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = '{"error": "something"}'
            
            result = vlm.name_heroes("dummy.png")
            
            assert result == []

    def test_name_heroes_converts_to_string(self):
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = '[1, 2, 3]'
            
            result = vlm.name_heroes("dummy.png")
            
            assert result == ["1", "2", "3"]


class TestJointTopology:
    def test_joint_topology_returns_dict(self):
        parts = [{"name": "base", "centroid": [0, 0, 0]}]
        expected_result = {"base": "base", "joints": []}
        
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = json.dumps(expected_result)
            
            result = vlm.joint_topology(parts)
            
            assert result == expected_result
            mock_chat.assert_called_once()

    def test_joint_topology_returns_empty_dict_on_invalid_json(self):
        parts = [{"name": "base"}]
        
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = "Invalid JSON"
            
            result = vlm.joint_topology(parts)
            
            assert result == {}

    def test_joint_topology_returns_empty_dict_on_list(self):
        parts = [{"name": "base"}]
        
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = '["not", "a", "dict"]'
            
            result = vlm.joint_topology(parts)
            
            assert result == {}

    def test_joint_topology_includes_legend_in_prompt(self):
        parts = [{"name": "part1"}]
        legend = {"part1": [255, 0, 0]}
        
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = '{"base": "part1", "joints": []}'
            
            vlm.joint_topology(parts, legend=legend)
            
            args, kwargs = mock_chat.call_args
            prompt = args[0]
            assert "part1: RGB(255, 0, 0)" in prompt

    def test_joint_topology_includes_parts_in_prompt(self):
        parts = [{"name": "part1", "centroid": [1, 2, 3]}]
        
        with patch.object(vlm, 'chat') as mock_chat:
            mock_chat.return_value = '{"base": "part1", "joints": []}'
            
            vlm.joint_topology(parts)
            
            args, kwargs = mock_chat.call_args
            prompt = args[0]
            assert "part1" in prompt
            assert "centroid" in prompt