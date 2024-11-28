class ClassRegistry:
  """
  A class to act as a registry for storing classes with associated names.
  """
  def __init__(self):
    self._registry = {}

  def add_to_registry(self, name):
    """
    A decorator to register a class with the given name.

    Args:
      name (str): The name to register the class under.

    Returns:
      decorator: The class decorator.
    """
    def decorator(cls):
      self._registry[name] = cls
      return cls
    return decorator
  
  def __getitem__(self, name):
    return self._registry[name]
  
  def get(self, name, default=None):
    return self._registry.get(name, default)