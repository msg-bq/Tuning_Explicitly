import re
from typing import Union, Set, List

from utils.ExtraNameSpace import KnowledgeExtractionNameSpace


@KnowledgeExtractionNameSpace.register("Default")
def extract_knowledge_texts(cls, rationale: str) -> Union[Set[str], List[str]]:
    knowledge_pattern = re.compile(r"<Begin>(.+?)</End>")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k.strip() for k in knowledge_texts if len(k.split()) > 2 and k.strip() != '']

    return knowledge_texts


@KnowledgeExtractionNameSpace.register("category_prompt")
def extract_knowledge_texts(cls, rationale: str) -> Union[Set[str], List[str]]:
    knowledge_pattern = re.compile(r"(we|We)\s+(have|retrieve)\s+\"(.+?)\"[.,;:?!]")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k[2].strip() for k in knowledge_texts]

    return knowledge_texts


@KnowledgeExtractionNameSpace.register("lang8")
def extract_knowledge_texts(cls, rationale: str) -> Union[Set[str], List[str]]:
    knowledge_pattern = re.compile(r"(we|We)\s+(have|retrieve)\s+\"(.+?)\"[.,;:?!]")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k[2].strip() for k in knowledge_texts]

    return knowledge_texts
