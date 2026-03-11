class BaseController:
    """Unified interface for all controllers."""
    
    def __init__(self):
        pass
        
    def compute_action(self, current_state, reference_state):
        raise NotImplementedError("Subclasses must implement compute_action()")
