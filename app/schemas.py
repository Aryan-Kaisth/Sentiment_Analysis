from pydantic import BaseModel, Field
from typing import Annotated

class SentimentText(BaseModel):
    Text: Annotated[
        str, 
        Field(
            ...,
            description="Text for which you want to check the sentiment",
            min_length=5,
            examples=[
                'I really enjoyed using this product — it’s reliable, looks great, and makes my daily routine much easier.',
                'I was disappointed with the service; it took too long to get help, and the staff didn’t seem very interested in solving my problem.'
            ]
        )
    ]
