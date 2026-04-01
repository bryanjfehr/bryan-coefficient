import outlines
from pydantic import BaseModel

class Foo(BaseModel):
    bar: str

print("generator hasattr json:", hasattr(outlines.generator, 'json'))
