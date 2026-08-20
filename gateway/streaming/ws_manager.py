"""
WebSocketManager: Full-duplex real-time connection manager for AETHERIS-Zero.
Handles client connection pooling, telemetry broadcasting, heartbeat pings,
and client-initiated command dispatches (OpenADR DR triggers, malicious fault injections).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("aetheris.streaming.ws_manager")


class ConnectionManager:
    """
    Manages active WebSocket connections and routes bidirectional messages
    between the 3D digital twin dashboard and the underlying simulation engine.
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket client."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected client."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(self.active_connections)}")

    @property
    def client_count(self) -> int:
        return len(self.active_connections)

    async def send_personal_message(self, message: Union[Dict[str, Any], str], websocket: WebSocket) -> None:
        """Send a JSON or string message to a specific client."""
        try:
            if isinstance(message, dict):
                await websocket.send_json(message)
            else:
                await websocket.send_text(str(message))
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)

    async def broadcast(self, message: Union[Dict[str, Any], str]) -> None:
        """Broadcast a telemetry frame or event notification to all active clients."""
        if not self.active_connections:
            return

        json_str = json.dumps(message) if isinstance(message, dict) else str(message)
        dead_connections: List[WebSocket] = []

        async with self._lock:
            for connection in list(self.active_connections):
                try:
                    await connection.send_text(json_str)
                except Exception as exc:
                    logger.debug(f"Error broadcasting to client, removing: {exc}")
                    dead_connections.append(connection)

            for dead in dead_connections:
                self.active_connections.discard(dead)

    async def handle_client_command(
        self,
        command_payload: Dict[str, Any],
        command_handlers: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming command from a dashboard client.
        Supported action types:
          - TRIGGER_OPENADR_EVENT
          - INJECT_MALICIOUS_SETPOINT
          - INJECT_DWELL_ATTACK
          - TOGGLE_SHADOW_MODE
          - RESET_SIMULATION
          - PING
        """
        action = command_payload.get("action") or command_payload.get("type", "UNKNOWN")
        logger.info(f"Received client command action: {action}")

        if action == "PING":
            return {"status": "PONG", "timestamp": command_payload.get("timestamp")}

        if command_handlers and action in command_handlers:
            try:
                handler = command_handlers[action]
                result = handler(command_payload.get("params", command_payload))
                if asyncio.iscoroutine(result):
                    result = await result
                return {"status": "SUCCESS", "action": action, "result": result}
            except Exception as e:
                logger.error(f"Error executing command {action}: {e}")
                return {"status": "ERROR", "action": action, "error": str(e)}

        return {"status": "UNHANDLED", "action": action, "message": f"No handler registered for {action}"}


# Singleton manager instance
ws_manager = ConnectionManager()
