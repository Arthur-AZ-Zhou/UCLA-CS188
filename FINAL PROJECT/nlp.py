from sentence_transformers import SentenceTransformer
import numpy as np
import random

model = SentenceTransformer("all-MiniLM-L6-v2")

# Define magnitude templates and their corresponding ranges
MAGNITUDE_TEMPLATES = {
    "small": ["small", "little", "slight", "bit", "tiny bit"],
    "medium": ["normal", "regular", "standard", "usual"],
    "large": ["large", "big", "way", "far", "much"],
}

# Define magnitude ranges with mean and standard deviation
MAGNITUDE_RANGES = {
    "small": (0.1, 0.05),   # mean=0.2, std=0.1
    "medium": (0.5, 0.1),  # mean=0.5, std=0.1
    "large": (1.0, 0.1),   # mean=1.0, std=0.1
}

# Define command templates and their corresponding actions
COMMAND_TEMPLATES = {
    "move left": ["move left", "go left", "move to the left"],
    "move right": ["move right", "go right", "move to the right"],
    "move forward": ["move forward", "go forward", "move ahead"],
    "move back": ["move back", "go back", "move backward"],
    "move up": ["move up", "go up", "move upward"],
    "move down": ["move down", "go down", "move downward"],
    "gripper open": ["open gripper", "open hand", "release grip", "release", "let go"],
    "gripper close": ["close gripper", "close hand", "grip", "grab"],
    "lift green": ["lift green cube", "pick up green", "grab green"],
    "lift red": ["lift red cube", "pick up red", "grab red"],
    "stack": ["stack cubes", "stack them", "put one on top"],
    "reset": ["reset", "restart", "start over"],
    "break": ["break", "exit", "quit", "stop"]
}

def get_command_embedding(text):
    """Get embedding for a single text."""
    return model.encode(text)

def get_template_embeddings():
    """Get embeddings for all command templates."""
    all_templates = []
    template_to_command = {}
    
    for command, templates in COMMAND_TEMPLATES.items():
        for template in templates:
            all_templates.append(template)
            template_to_command[template] = command
    
    embeddings = model.encode(all_templates)
    return embeddings, all_templates, template_to_command

def get_magnitude_embedding(text):
    """Get embedding for a single text."""
    return model.encode(text)

def get_magnitude_templates_embeddings():
    """Get embeddings for all magnitude templates."""
    all_templates = []
    template_to_magnitude = {}
    
    for magnitude, templates in MAGNITUDE_TEMPLATES.items():
        for template in templates:
            all_templates.append(template)
            template_to_magnitude[template] = magnitude
    
    embeddings = model.encode(all_templates)
    return embeddings, all_templates, template_to_magnitude

def get_random_magnitude(magnitude_type):
    """Get a random magnitude from the specified range using normal distribution."""
    mean, std = MAGNITUDE_RANGES[magnitude_type]
    # Ensure the magnitude is positive
    magnitude = abs(np.random.normal(mean, std))
    return magnitude

def process_command(user_input):
    """
    Process user input and return a tuple: (command, magnitude)
    If no matching command or magnitude found, returns (None, None)
    """
    user_input = user_input.lower()
    
    # Get magnitude templates embeddings
    magnitude_embeddings, magnitude_templates, template_to_magnitude = get_magnitude_templates_embeddings()
    user_embedding = get_magnitude_embedding(user_input)
    
    # Find the most similar magnitude template
    magnitude_similarities = np.dot(magnitude_embeddings, user_embedding) / (
        np.linalg.norm(magnitude_embeddings, axis=1) * np.linalg.norm(user_embedding)
    )
    
    most_similar_magnitude_idx = np.argmax(magnitude_similarities)
    magnitude_similarity_score = magnitude_similarities[most_similar_magnitude_idx]
    
    # If we found a magnitude term with high confidence, use it
    if magnitude_similarity_score > 0.7:
        magnitude_type = template_to_magnitude[magnitude_templates[most_similar_magnitude_idx]]
        magnitude = get_random_magnitude(magnitude_type)
    else:
        # Default to medium magnitude if no specific magnitude is detected
        magnitude = get_random_magnitude("medium")
    
    # Get command templates embeddings
    template_embeddings, templates, template_to_command = get_template_embeddings()
    user_embedding = get_command_embedding(user_input)
    
    # Cosine similarity for command matching
    similarities = np.dot(template_embeddings, user_embedding) / (
        np.linalg.norm(template_embeddings, axis=1) * np.linalg.norm(user_embedding)
    )
    
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[most_similar_idx]

    if similarity_score > 0.7:
        return template_to_command[templates[most_similar_idx]], magnitude

    return None, None

