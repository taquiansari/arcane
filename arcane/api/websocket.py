"""WebSocket handler for real-time streaming of research progress."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from arcane.memory.session import get_session_manager
from arcane.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections per session."""

    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        if session_id not in self.active:
            self.active[session_id] = []
        self.active[session_id].append(websocket)
        logger.info("ws_connected", session_id=session_id)

    def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        if session_id in self.active:
            self.active[session_id] = [
                ws for ws in self.active[session_id] if ws != websocket
            ]
            if not self.active[session_id]:
                del self.active[session_id]
        logger.info("ws_disconnected", session_id=session_id)

    async def send_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Send an event to all connected clients for a session."""
        if session_id not in self.active:
            return

        message = json.dumps({"type": event_type, "data": data})

        dead_connections = []
        for ws in self.active[session_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(ws, session_id)


# Global connection manager
ws_manager = ConnectionManager()


@router.websocket("/ws/research/{session_id}")
async def research_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for streaming research progress.

    Clients connect here to receive real-time updates during
    a research session. Events include:
    - status: Pipeline stage changes
    - progress: Search/retrieval progress
    - draft: New draft report available
    - critique: Critique evaluation results
    - final: Completed research report
    - error: Error notifications
    """
    await ws_manager.connect(websocket, session_id)

    try:
        manager = get_session_manager()

        # Send current status immediately
        session = manager.get_session(session_id)
        if session:
            await websocket.send_text(json.dumps({
                "type": "status",
                "data": {
                    "stage": session["status"],
                    "message": f"Connected to session {session_id}",
                },
            }))

        # Poll for updates and stream them to the client
        last_status = session["status"] if session else ""
        while True:
            try:
                # Check for incoming messages (e.g., feedback)
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=2.0,
                    )
                    msg = json.loads(data)
                    if msg.get("type") == "feedback":
                        manager.update_session(session_id, {
                            "human_feedback": msg.get("data", {}).get("feedback", ""),
                        })
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "data": {"message": "Feedback received"},
                        }))
                except asyncio.TimeoutError:
                    pass

                # Check for status changes
                session = manager.get_session(session_id)
                if session and session["status"] != last_status:
                    last_status = session["status"]

                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "data": {
                            "stage": last_status,
                            "message": f"Status changed to: {last_status}",
                        },
                    }))

                    # Send final result when complete
                    if last_status == "complete" and session.get("result"):
                        result = session["result"]
                        await websocket.send_text(json.dumps({
                            "type": "final",
                            "data": {
                                "report": result.get("final_report", ""),
                                "citations": result.get("citations", []),
                                "critique_score": result.get("critique_score"),
                                "revision_count": result.get("revision_count"),
                            },
                        }))
                        break

                    if last_status in ("failed", "cancelled"):
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "data": {
                                "message": session.get("error", "Session ended"),
                            },
                        }))
                        break

                await asyncio.sleep(1)

            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, session_id)
