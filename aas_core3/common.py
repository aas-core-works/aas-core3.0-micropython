def assert_never(value):

    assert False, f"Unhandled value: {value} ({type(value).__name__})"
