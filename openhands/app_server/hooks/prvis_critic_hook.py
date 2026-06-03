"""
prvis-ai dual-brain critic hook for OpenHands 1.7
Integrates as an OpenHands hook via .openhands/hooks.json

Brain B (devstral:24b) reviews every agent action before execution.
Rejects trigger replanning. Max 3 rounds before Telegram escalation.

Install:
  1. Copy this file to openhands/app_server/hooks/prvis_critic_hook.py
  2. Add to .openhands/hooks.json in your repo:
     {
       "hooks": [{"type": "prvis_critic", "config": {}}]
     }
"""

import os
import json
import asyncio
import httpx
from typing import Any

# ── Config from env ───────────────────────────────────────────────────────────
BRAIN_B_URL      = os.environ.get("BRAIN_B_URL", "http://host-gateway:11434")
BRAIN_B_MODEL    = os.environ.get("BRAIN_B_MODEL", "devstral:24b")
CRITIC_MAX_ROUNDS = int(os.environ.get("CRITIC_MAX_ROUNDS", "3"))
CRITIC_TIMEOUT   = int(os.environ.get("CRITIC_TIMEOUT", "120"))
TELEGRAM_URL     = os.environ.get("TELEGRAM_GATEWAY_URL", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Actions that skip critic (safe, non-destructive)
SKIP_ACTION_TYPES = {
    "MessageAction",
    "AgentFinishAction",
    "AgentRejectAction",
    "NullAction",
    "ChangeAgentStateAction",
}

# Destructive actions that always escalate
ESCALATE_ACTION_TYPES = {
    "FileDeleteAction",
}

CRITIC_SYSTEM = """You are Brain B — the prvis critic LLM. Brain A proposed an action.
Review it for safety, correctness, and infrastructure compliance.

Infrastructure rules to enforce:
- SSH must always go through deploy-vm (192.168.15.133) first
- Never direct SSH to Hetzner (100.65.209.30) from agent sandbox
- Never delete files without explicit user instruction
- Never expose secrets in commands
- Git operations must go through deploy-vm

Return ONLY valid JSON:
{
  "verdict": "APPROVE" | "REJECT" | "ESCALATE",
  "reason": "one sentence",
  "suggestion": "specific fix if rejecting, null if approving or escalating"
}"""


async def call_critic(action_type: str, action_args: dict, goal: str, round_num: int) -> dict:
    """Call Brain B to review the action."""
    prompt = (
        f"Goal: {goal}\n"
        f"Critic round: {round_num}/{CRITIC_MAX_ROUNDS}\n"
        f"Action type: {action_type}\n"
        f"Action args: {json.dumps(action_args, indent=2)[:1000]}\n\n"
        f"Review this. Return JSON only."
    )
    try:
        async with httpx.AsyncClient(timeout=CRITIC_TIMEOUT) as c:
            r = await c.post(
                f"{BRAIN_B_URL}/api/generate",
                json={
                    "model": BRAIN_B_MODEL,
                    "prompt": prompt,
                    "system": CRITIC_SYSTEM,
                    "stream": False,
                    "format": "json",
                    "options": {"think": False, "temperature": 0.1}
                }
            )
            r.raise_for_status()
            raw = r.json().get("response", "{}")
            return json.loads(raw)
    except Exception as e:
        # Critic unavailable — approve to not block
        return {
            "verdict": "APPROVE",
            "reason": f"Critic unavailable ({type(e).__name__}) — auto-approved",
            "suggestion": None
        }


async def notify_escalation(message: str) -> None:
    """Send escalation alert via OpenClaw/Telegram."""
    if not TELEGRAM_URL or not TELEGRAM_CHAT_ID:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                f"{TELEGRAM_URL}/send",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": f"🚨 *prvis-agent escalation*\n\n{message}",
                    "parse_mode": "Markdown"
                }
            )
    except Exception:
        pass


class PrvisCriticHook:
    """
    OpenHands 1.7 hook that intercepts agent actions for Brain B review.
    
    Add to .openhands/hooks.json:
    {
      "hooks": [
        {
          "type": "prvis_critic",
          "config": {
            "brain_b_url": "http://host-gateway:11434",
            "brain_b_model": "devstral:24b",
            "max_rounds": 3
          }
        }
      ]
    }
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._round_counts: dict[str, int] = {}

    async def on_action(
        self,
        conversation_id: str,
        action_type: str,
        action_args: dict,
        goal: str = "",
        **kwargs: Any,
    ) -> dict:
        """
        Called before every agent action.
        Returns modified action args or raises to block.
        """
        # Skip safe actions
        if action_type in SKIP_ACTION_TYPES:
            return action_args

        # Always escalate destructive actions
        if action_type in ESCALATE_ACTION_TYPES:
            msg = f"Destructive action blocked: {action_type}\nArgs: {json.dumps(action_args)[:500]}"
            await notify_escalation(msg)
            raise ValueError(f"ESCALATED: {action_type} requires explicit user confirmation")

        # Track critic rounds per conversation
        round_num = self._round_counts.get(conversation_id, 0) + 1
        self._round_counts[conversation_id] = round_num

        verdict = await call_critic(action_type, action_args, goal, round_num)
        decision = verdict.get("verdict", "APPROVE")
        reason = verdict.get("reason", "")
        suggestion = verdict.get("suggestion")

        if decision == "APPROVE":
            # Reset round counter on approval
            self._round_counts[conversation_id] = 0
            return action_args

        elif decision == "ESCALATE":
            msg = (
                f"Brain B escalated action:\n"
                f"Type: {action_type}\n"
                f"Reason: {reason}\n"
                f"Args: {json.dumps(action_args)[:500]}"
            )
            await notify_escalation(msg)
            raise ValueError(f"ESCALATED: {reason}")

        else:  # REJECT
            if round_num >= CRITIC_MAX_ROUNDS:
                msg = (
                    f"Max critic rounds ({CRITIC_MAX_ROUNDS}) reached.\n"
                    f"Last rejection: {reason}\n"
                    f"Action: {action_type}"
                )
                await notify_escalation(msg)
                self._round_counts[conversation_id] = 0
                raise ValueError(f"MAX_ROUNDS_EXCEEDED: {reason}")

            # Inject critique into action args for Brain A to replan
            action_args["_critic_feedback"] = (
                f"Round {round_num} critic rejection: {reason}. "
                f"Suggestion: {suggestion or 'reconsider approach'}"
            )
            return action_args

    async def on_conversation_end(self, conversation_id: str, **kwargs: Any) -> None:
        """Clean up round tracking on conversation end."""
        self._round_counts.pop(conversation_id, None)


# ── Hook registration ─────────────────────────────────────────────────────────
def create_hook(config: dict) -> PrvisCriticHook:
    """Factory function called by OpenHands hook loader."""
    return PrvisCriticHook(config)
