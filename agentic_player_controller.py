from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx
import websockets
import yaml


try:
	from mcp_agent.agents.agent import Agent
	from mcp_agent.app import MCPApp
	from mcp_agent.config import MCPServerSettings, MCPSettings, OpenAISettings, Settings
	from mcp_agent.workflows.llm.augmented_llm import RequestParams
	from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
except ImportError:
	Agent = MCPApp = MCPServerSettings = MCPSettings = OpenAISettings = RequestParams = OpenAIAugmentedLLM = Settings = None


ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "player_prompt.md"
CONFIG_PATH = ROOT / "runner.yaml"
ACTION_SCHEMA = {
	"farmer": {"op": "PASS", "args": []},
	"hands": [],
	"market": [],
}
LLM_REQUEST_SEMAPHORE: asyncio.Semaphore | None = None


@dataclass
class RunnerConfig:
	backend_url: str
	mcp_url: str
	max_concurrent_llm_requests: int
	default_context_tokens: int
	default_handoff_at: float
	default_retry_attempts: int
	health_check_seconds: float
	max_respawns: int
	request_timeout_seconds: float
	episode_config: dict[str, Any]
	players: dict[int, "PlayerConfig"]


@dataclass
class PlayerConfig:
	prompt_file: str
	model: str
	base_url: str
	api_key: str | None
	conversation_mode: str
	handoff_at: float
	retry_attempts: int
	temperature: float
	max_output_tokens: int


def player_config(raw: dict[str, Any], defaults: dict[str, Any]) -> PlayerConfig:
	merged = {**defaults, **raw}
	mode = str(merged.get("conversation_mode", "new_agent_at_threshold"))
	if mode not in {"persistent", "new_agent_at_threshold", "new_agent_every_turn"}:
		raise ValueError(f"unsupported conversation_mode: {mode}")
	return PlayerConfig(
		prompt_file=str(merged.get("prompt_file", "player_prompt.md")),
		model=str(merged["model"]),
		base_url=str(merged["base_url"]).rstrip("/"),
		api_key=merged.get("api_key") or None,
		conversation_mode=mode,
		handoff_at=float(merged.get("handoff_at", 0.5)),
		retry_attempts=int(merged.get("retry_attempts", 3)),
		temperature=float(merged.get("temperature", 0.2)),
		max_output_tokens=int(merged.get("max_output_tokens", 1024)),
	)


def load_config(path: Path) -> RunnerConfig:
	raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
	llm = raw.get("llm", {})
	monitor = raw.get("monitoring", {})
	defaults = {
		"model": llm.get("model", "kaggriculture"),
		"base_url": llm.get("base_url", "http://localhost:8001/v1"),
		"api_key": llm.get("api_key"),
		"handoff_at": llm.get("handoff_at", 0.5),
		"retry_attempts": llm.get("retry_attempts", 3),
		"temperature": llm.get("temperature", 0.2),
		"max_output_tokens": llm.get("max_output_tokens", 1024),
	}
	players_raw = raw.get("players", {})
	return RunnerConfig(
		backend_url=str(raw.get("backend_url", "http://localhost:8000")).rstrip("/"),
		mcp_url=str(raw.get("mcp_server", {}).get("url", "http://localhost:8080/mcp")).rstrip("/"),
		max_concurrent_llm_requests=max(1, int(llm.get("max_concurrent_requests", 1))),
		default_context_tokens=int(llm.get("max_context_tokens", 32768)),
		default_handoff_at=float(llm.get("handoff_at", 0.5)),
		default_retry_attempts=int(llm.get("retry_attempts", 3)),
		health_check_seconds=float(monitor.get("health_check_seconds", 180)),
		max_respawns=int(monitor.get("max_respawns", 3)),
		request_timeout_seconds=float(raw.get("request_timeout_seconds", 120)),
		episode_config=dict(raw.get("episode", {})),
		players={player: player_config(players_raw.get(str(player), {}), defaults) for player in range(2)},
	)


def websocket_url(http_url: str, episode_id: UUID, token: str) -> str:
	parts = urlsplit(http_url)
	scheme = "wss" if parts.scheme == "https" else "ws"
	return urlunsplit((scheme, parts.netloc, f"/ws/episodes/{episode_id}", "token=" + token, ""))


def extract_json(text: str) -> dict[str, Any]:
	fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
	candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
	if not candidate:
		return ACTION_SCHEMA.copy()
	value = json.loads(candidate)
	if not isinstance(value, dict):
		raise ValueError("model response was not a JSON object")
	return value


def normalize_action(value: dict[str, Any]) -> dict[str, Any]:
	action = {"farmer": value.get("farmer", {"op": "PASS", "args": []}), "hands": value.get("hands", []), "market": value.get("market", [])}
	if not isinstance(action["farmer"], dict) or not isinstance(action["hands"], list) or not isinstance(action["market"], list):
		raise ValueError("action fields have invalid types")
	return action


def estimate_tokens(text: str) -> int:
	try:
		import tiktoken

		return len(tiktoken.get_encoding("cl100k_base").encode(text))
	except Exception:
		return max(1, len(text) // 4)


class PlayerLog:
	def __init__(self, player: int, episode_id: str):
		self.path = ROOT / f"player_{player}_{episode_id}.log"
		self._file = self.path.open("a", encoding="utf-8")

	def write(self, event: str, payload: Any) -> None:
		self._file.write(json.dumps({"time": time.time(), "event": event, "data": payload}, ensure_ascii=False, default=str) + "\n")
		self._file.flush()

	def close(self) -> None:
		self._file.close()


class PlayerRunner:
	def __init__(self, player: int, episode_id: UUID, token: str, config: RunnerConfig, prompt: str, app: Any):
		self.player = player
		self.episode_id = episode_id
		self.token = token
		self.config = config
		self.player_config = config.players[player]
		self.prompt = prompt
		self.app = app
		self.log = PlayerLog(player, str(episode_id))
		self.used_tokens = 0
		self.context_tokens = getattr(self.player_config, "reported_context_tokens", config.default_context_tokens)
		self.respawns = 0
		self.memory = "No previous agent summary is available."
		self.agent: Any = None
		self.llm: Any = None
		self.last_observation: dict[str, Any] | None = None
		self.last_status: dict[str, Any] | None = None

	def instruction(self) -> str:
		return self.prompt + f"\nEpisode_Id={self.episode_id}\nplayer_Id={self.player}\n\nOperational memory from the previous agent:\n{self.memory}\n"

	async def create_agent(self) -> tuple[Any, Any]:
		agent = Agent(name=f"player_{self.player}", instruction=self.instruction(), server_names=["kaggriculture"])
		await agent.initialize()
		llm = await agent.attach_llm(OpenAIAugmentedLLM)
		return agent, llm

	async def generate(self, observation: dict[str, Any]) -> tuple[dict[str, Any], int]:
		if Agent is None or Settings is None:
			raise RuntimeError("Install mcp-agent[openai] in the interpreter running this file")
		request = "Analyze this observation and return only a JSON PlayerAction. Do not call tools. Observation:\n" + json.dumps(observation, separators=(",", ":"))
		transient = self.player_config.conversation_mode == "new_agent_every_turn"
		agent = self.agent
		llm = self.llm
		if transient or agent is None:
			agent, llm = await self.create_agent()
		try:
			if LLM_REQUEST_SEMAPHORE is None:
				response = await llm.generate_str(message=request, request_params=self.request_params())
			else:
				async with LLM_REQUEST_SEMAPHORE:
					response = await llm.generate_str(message=request, request_params=self.request_params())
			self.log.write("response", response)
			used = estimate_tokens(request + response)
			action = normalize_action(extract_json(response))
			self.memory = json.dumps({"last_step": observation.get("step"), "last_action": action, "pending_plan": "Continue from the latest observation and choose the next valid action."}, separators=(",", ":"))
			return action, used
		finally:
			if transient:
				await agent.shutdown()

	def request_params(self) -> Any:
		return RequestParams(model=self.player_config.model, maxTokens=self.player_config.max_output_tokens, temperature=self.player_config.temperature, max_iterations=1, use_history=self.player_config.conversation_mode == "persistent", tool_filter={})

	def handoff_summary(self, observation: dict[str, Any]) -> str:
		return json.dumps({"episode_id": str(self.episode_id), "player_id": self.player, "pending_plan": "Continue playing from the latest observation.", "memory": self.memory, "latest_step": observation.get("step")}, separators=(",", ":"))

	async def run(self) -> None:
		async with self.app.run():
			await self._run_socket()

	async def _run_socket(self) -> None:
		uri = websocket_url(self.config.backend_url, self.episode_id, self.token)
		try:
			async with websockets.connect(uri, open_timeout=self.config.request_timeout_seconds) as socket:
				if self.player_config.conversation_mode != "new_agent_every_turn":
					self.agent, self.llm = await self.create_agent()
				while True:
					raw = json.loads(await socket.recv())
					message_type = raw.get("type")
					if message_type == "status":
						self.last_status = raw["status"]
						if self.last_status["status"] == "finished":
							return
					elif message_type == "observation":
						observation = raw["observation"]
						self.last_observation = observation
						self.log.write("observation", observation)
						if self.player_config.conversation_mode == "new_agent_at_threshold" and self.used_tokens >= self.context_tokens * self.player_config.handoff_at:
							self.log.write("handoff", {"used_tokens": self.used_tokens, "summary": self.handoff_summary(observation)})
							await self.agent.shutdown()
							self.agent, self.llm = await self.create_agent()
							self.used_tokens = 0
						action = None
						last_error = None
						for attempt in range(1, self.player_config.retry_attempts + 1):
							try:
								action, used = await self.generate(observation)
								break
							except Exception as exc:
								last_error = exc
								if attempt == self.player_config.retry_attempts:
									raise RuntimeError(f"player {self.player} failed after {attempt} attempts: {exc}") from exc
						if action is None:
							raise RuntimeError(str(last_error))
						self.used_tokens += used
						await socket.send(json.dumps({"type": "action", "action": action}))
					elif message_type == "error":
						raise RuntimeError(raw.get("error", "backend websocket error"))
		except Exception as exc:
			self.log.write("error", str(exc))
			raise
		finally:
			if self.agent is not None and self.agent.initialized:
				await self.agent.shutdown()
			self.log.close()


async def create_episode(config: RunnerConfig) -> tuple[UUID, list[str]]:
	async with httpx.AsyncClient(base_url=config.backend_url, timeout=config.request_timeout_seconds) as client:
		response = await client.post("/api/episodes", json=config.episode_config)
		response.raise_for_status()
		payload = response.json()
	return UUID(payload["episode_id"]), payload["player_tokens"]


async def discover_context_limit(player_config: PlayerConfig, fallback: int) -> int:
	async with httpx.AsyncClient(base_url=player_config.base_url, timeout=30) as client:
		response = await client.get("/models")
		response.raise_for_status()
		payload = response.json()
	models = payload.get("models", payload.get("data", []))
	selected = next((item for item in models if item.get("id") == player_config.model or item.get("model") == player_config.model or item.get("name") == player_config.model), models[0] if models else {})
	meta = selected.get("meta", {})
	for key in ("n_ctx", "context_length", "max_context_length"):
		value = selected.get(key, meta.get(key))
		if isinstance(value, int) and value > 0:
			return value
	return fallback


async def monitor(config: RunnerConfig, episode_id: UUID, runners: list[PlayerRunner], tasks: list[asyncio.Task[Any]]) -> None:
	while True:
		done, _ = await asyncio.wait(tasks, timeout=config.health_check_seconds)
		if len(done) == len(tasks):
			await asyncio.gather(*tasks)
			return
		try:
			async with httpx.AsyncClient(base_url=config.backend_url, timeout=config.request_timeout_seconds) as client:
				response = await client.get(f"/api/episodes/{episode_id}")
				response.raise_for_status()
				status = response.json()
			if status["status"] == "finished":
				return
			for runner, task in zip(runners, tasks):
				if task.done() and runner.respawns < config.max_respawns:
					runner.respawns += 1
					runner.log = PlayerLog(runner.player, str(episode_id))
					tasks[runners.index(runner)] = asyncio.create_task(runner.run())
					runner.log.write("respawn", {"attempt": runner.respawns, "status": status})
				elif task.done():
					runner.log.write("error", "maximum respawn attempts exceeded")
					raise RuntimeError(f"player {runner.player} exceeded respawn attempts")
		except Exception as exc:
			for runner in runners:
				runner.log.write("error", str(exc))
			raise


async def main(config_path: Path) -> None:
	if MCPApp is None:
		raise SystemExit("Install mcp-agent[openai] in the interpreter running this file")
	config = load_config(config_path)
	episode_id, tokens = await create_episode(config)
	os.chdir(ROOT)
	global LLM_REQUEST_SEMAPHORE
	LLM_REQUEST_SEMAPHORE = asyncio.Semaphore(config.max_concurrent_llm_requests)
	runners = []
	for player in range(2):
		player_llm = config.players[player]
		config.players[player].reported_context_tokens = await discover_context_limit(player_llm, config.default_context_tokens)
		settings = Settings(
			execution_engine="asyncio",
			mcp=MCPSettings(servers={"kaggriculture": MCPServerSettings(transport="streamable_http", url=config.mcp_url)}),
			openai=OpenAISettings(api_key=player_llm.api_key or "not-required", base_url=player_llm.base_url, default_model=player_llm.model, reasoning_effort="none"),
		)
		app = MCPApp(name=f"kaggriculture_player_{player}", settings=settings)
		prompt = (ROOT / player_llm.prompt_file).read_text(encoding="utf-8")
		runners.append(PlayerRunner(player, episode_id, tokens[player], config, prompt, app))
	tasks = [asyncio.create_task(runner.run()) for runner in runners]
	monitor_task = asyncio.create_task(monitor(config, episode_id, runners, tasks))
	try:
		await monitor_task
		await asyncio.gather(*tasks)
	finally:
		monitor_task.cancel()
		await asyncio.gather(monitor_task, return_exceptions=True)
		for task in tasks:
			if not task.done():
				task.cancel()
		await asyncio.gather(*tasks, return_exceptions=True)


def main_cli() -> None:
	parser = argparse.ArgumentParser(description="Run two MCP-agent Kaggriculture players")
	parser.add_argument("--config", type=Path, default=CONFIG_PATH)
	args = parser.parse_args()
	logging.basicConfig(level=logging.INFO)
	asyncio.run(main(args.config))


if __name__ == "__main__":
	main_cli()
