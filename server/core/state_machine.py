from enum import Enum
import time
from typing import Optional, Callable, Awaitable


class AgentState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class StateMachine:
    """
    Manages the conversational turn-taking lifecycle and handles barge-in interrupts.
    """

    def __init__(self, on_state_change: Optional[Callable[[AgentState, AgentState], Awaitable[None]]] = None):
        self.current_state: AgentState = AgentState.IDLE
        self.state_entered_at: float = time.time()
        self.generation_id: int = 0  # Monotonically increasing ID to invalidate stale audio tasks
        self.on_state_change = on_state_change

    async def transition_to(self, new_state: AgentState):
        """Transition to a new state and notify listener."""
        if self.current_state == new_state:
            return

        old_state = self.current_state
        self.current_state = new_state
        self.state_entered_at = time.time()

        if new_state == AgentState.INTERRUPTED:
            # Invalidate previous generation
            self.generation_id += 1

        if self.on_state_change:
            await self.on_state_change(old_state, new_state)

    def is_speaking(self) -> bool:
        return self.current_state == AgentState.SPEAKING

    def get_duration_in_current_state(self) -> float:
        return time.time() - self.state_entered_at
