from typing import Annotated
from pydantic import BaseModel, Field    

class RagQueryRequest(BaseModel):
    query: Annotated[
        str,
        Field(..., description = "User's latest question", examples = ["What is an algorithm?"])
    ]
    
    semester: Annotated[
        int,
        Field(..., ge = 1, le = 8, description = "Semester number", examples = [1, 2])
    ]

    subject: Annotated[
        str,
        Field(..., description = "subject name", examples = ["aoa", "os"])
    ]

    unit: Annotated[
        int,
        Field(..., ge = 1, le = 7, description = "unit number of the subject", examples = [1, 2])
    ]

    thread_id: Annotated[
        str,
        Field(..., description = "thread id of the current conversation")
    ]


class RenameThreadRequest(BaseModel):
    title: str