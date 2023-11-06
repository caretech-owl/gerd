from pydantic import BaseModel


class FeaturesConfig(BaseModel):
    fact_checking: bool
    return_source: bool
