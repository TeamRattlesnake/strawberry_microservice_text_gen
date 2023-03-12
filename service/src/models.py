from pydantic import BaseModel


class ResponseModel(BaseModel):
    result: str


class AddGroupModel(BaseModel):
    group_id: int
    texts: list[str]


class GenerateModel(BaseModel):
    group_id: int
    hint: str
