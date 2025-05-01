"""
Domain Utilities for Java Peer Review Training System.

This module provides centralized domain management to avoid duplication
across different components of the system.
"""

import logging
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Common Java domains with associated keywords
DOMAINS = {
    "student_management": ["student", "course", "enroll", "grade", "academic"],
    "file_processing": ["file", "read", "write", "path", "directory"],
    "data_validation": ["validate", "input", "check", "valid", "sanitize"],
    "calculation": ["calculate", "compute", "math", "formula", "result"],
    "inventory_system": ["inventory", "product", "stock", "item", "quantity"],
    "notification_service": ["notify", "message", "alert", "notification", "send"],
    "banking": ["account", "bank", "transaction", "balance", "deposit"],
    "e-commerce": ["cart", "product", "order", "payment", "customer"],
    "general_application": []  # Default domain
}

# List version for selection dropdowns
DOMAIN_LIST = list(DOMAINS.keys())

def infer_domain_from_code(code: str) -> str:
    """
    Infer the domain of the code based on class and variable names.
    
    Args:
        code: The Java code to analyze
        
    Returns:
        Inferred domain string
    """
    if not code:
        return "general_application"
        
    code_lower = code.lower()
    
    # Count domain-related terms
    domain_scores = {}
    for domain, terms in DOMAINS.items():
        # Skip the default domain which has no terms
        if not terms:
            continue
            
        score = sum(code_lower.count(term) for term in terms)
        domain_scores[domain] = score
    
    # Return the highest scoring domain, or a default
    if domain_scores:
        max_domain = max(domain_scores.items(), key=lambda x: x[1])
        if max_domain[1] > 0:
            logger.debug(f"Inferred domain '{max_domain[0]}' with score {max_domain[1]}")
            return max_domain[0]
    
    logger.debug("No specific domain detected, using general_application")
    return "general_application"

def get_random_domain() -> str:
    """
    Get a random domain for code generation.
    
    Returns:
        Random domain name
    """
    import random
    # Exclude general_application from random selection
    domains = [d for d in DOMAIN_LIST if d != "general_application"] 
    return random.choice(domains)

def update_state_with_domain(state, code: Optional[str] = None, domain: Optional[str] = None):
    """
    Update the workflow state with domain information.
    
    Args:
        state: The workflow state to update
        code: Optional code to infer domain from
        domain: Optional explicit domain to use
        
    Returns:
        Updated state with domain information
    """
    # If domain is explicitly provided, use it
    if domain and domain in DOMAINS:
        state.domain = domain
        logger.info(f"Using explicit domain: {domain}")
        return state
    
    # If there's already a domain in the state, keep it
    if hasattr(state, 'domain') and state.domain and state.domain in DOMAINS:
        logger.debug(f"Keeping existing domain: {state.domain}")
        return state
    
    # If code is provided, infer domain
    if code:
        inferred_domain = infer_domain_from_code(code)
        state.domain = inferred_domain
        logger.info(f"Inferred domain from code: {inferred_domain}")
        return state
    
    # If no code or domain provided, use a random domain
    random_domain = get_random_domain()
    state.domain = random_domain
    logger.info(f"Using random domain: {random_domain}")
    return state