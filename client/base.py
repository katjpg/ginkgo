from abc import ABC, abstractmethod

class BaseClient(ABC):
    """ 
    TODO: 
    
    - abstract base class for api clients
    - thinking of implementing (but currently undecided): 
        - def fetch() with @abstractmethod decorator 
    
    
    """
    pass

class RateLimiter():
    """ 
    TODO: 
    
    - a simple rate limiter to ensure min time between requests
    
    """
    pass