# Kaggriculture agent runner

This runner creates one backend episode, starts two `mcp-agent` players, and drives each player through the backend WebSocket. The shared MCP server is configured as a Streamable HTTP server in `runner.yaml`; the backend WebSocket remains the action channel required by the simulation engine.

## Run

Start the simulation backend and its MCP server, then start vLLM with an OpenAI-compatible endpoint. From the repository root:

```powershell
python kaggriculture_agentic_player\agentic_player_controller.py
```

Edit `runner.yaml` to change the backend, shared MCP URL, episode settings, monitoring interval, and LLM concurrency. Each player can independently set `model`, `base_url`, `api_key`, prompt file, temperature, output limit, retries, and conversation mode.

Conversation modes are `persistent` (one conversation), `new_agent_at_threshold` (replace it when the server-reported context budget reaches `handoff_at`), and `new_agent_every_turn` (fresh conversation for every observation). The runner reads the context limit from each provider's `/models` response; `max_context_tokens` is only a fallback when the server does not report one. `max_concurrent_requests` controls whether one or multiple provider requests may run at once.

Logs are written beside this file as `player_0_<episode-id>.log` and `player_1_<episode-id>.log`. They contain observations, model responses, handoff events, respawns, and errors.

At half of the configured context budget, the current agent is closed and a fresh agent receives the episode ID, player ID, latest observation, and a continuation instruction. The monitor checks episode status periodically and respawns stopped players up to three times.