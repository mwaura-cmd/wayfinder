from typing import Any
from execution.parser import AgentAction, ActionType
from execution.governor import IterationGovernor
from core.memory import WorkingMemory

# In a real app we might import Payload from telemetry, but avoiding circular/heavy deps here.
class StopCondition:
    name: str

    def check(self, action: AgentAction, governor: IterationGovernor, memory: WorkingMemory) -> bool:
        raise NotImplementedError

    def get_payload(self) -> dict:
        raise NotImplementedError

class FinalAnswerCondition(StopCondition):
    name = "FinalAnswerCondition"
    
    def check(self, action: AgentAction, governor: IterationGovernor, memory: WorkingMemory) -> bool:
        return action.type == ActionType.FINAL_ANSWER
        
    def get_payload(self) -> dict:
        return {"type": "run_complete", "summary": "Final answer reached"}

class MaxIterationsCondition(StopCondition):
    name = "MaxIterationsCondition"
    
    def check(self, action: AgentAction, governor: IterationGovernor, memory: WorkingMemory) -> bool:
        return governor.steps >= governor.max_steps
        
    def get_payload(self) -> dict:
        return {"type": "run_complete", "summary": "Max iterations reached"}

class ContextOverflowCondition(StopCondition):
    name = "ContextOverflowCondition"
    
    def check(self, action: AgentAction, governor: IterationGovernor, memory: WorkingMemory) -> bool:
        # Hardcoding the context limit here to avoid passing provider just for this check in the stub
        return memory.get_token_count(None) >= 1000000 
        
    def get_payload(self) -> dict:
        return {"type": "run_complete", "summary": "Context window exhausted"}

class ConsecutiveMalformedCondition(StopCondition):
    name = "ConsecutiveMalformedCondition"
    
    def check(self, action: AgentAction, governor: IterationGovernor, memory: WorkingMemory) -> bool:
        return governor.malformed_count >= 3
        
    def get_payload(self) -> dict:
        return {"type": "run_complete", "summary": "Loop aborted — 3 consecutive parse failures"}

class StopConditionRegistry:
    @classmethod
    def defaults(cls) -> list[StopCondition]:
        return [
            FinalAnswerCondition(),
            MaxIterationsCondition(),
            ContextOverflowCondition(),
            ConsecutiveMalformedCondition()
        ]
