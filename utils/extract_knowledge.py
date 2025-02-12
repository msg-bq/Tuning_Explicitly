import re
from typing import Union

from utils.ExtraNameSpace import KnowledgeExtractionNameSpace


@KnowledgeExtractionNameSpace.register("Default")
def extract_knowledge_texts(rationale: str) -> Union[set[str], list[str]]:
    knowledge_pattern = re.compile(r"<Begin>(.+?)</End>")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k.strip() for k in knowledge_texts if len(k.split()) > 2 and k.strip() != '']

    return knowledge_texts


@KnowledgeExtractionNameSpace.register("CLUTRR")
def extract_knowledge_texts(rationale: str) -> Union[set[str], list[str]]:
    knowledge_pattern = re.compile(r"(we|We)\s+(have|retrieve)\s+\"(.+?)\"[.,;:?!]")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k[2].strip() for k in knowledge_texts]

    return knowledge_texts


@KnowledgeExtractionNameSpace.register("lang8")
def extract_knowledge_texts(rationale: str) -> Union[set[str], list[str]]:
    knowledge_pattern = re.compile(r"(we|We)\s+(have|retrieve)\s+\"(.+?)\"[.,;:?!]")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k[2].strip() for k in knowledge_texts]

    return knowledge_texts


@KnowledgeExtractionNameSpace.register("SALAD")
def extract_knowledge_texts(rationale: str) -> Union[set[str], list[str]]:
    knowledge_pattern = re.compile(r"(we|We)\s+(have|retrieve)\s+\"(.+?)\"[.,;:?!]")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k[2].strip() for k in knowledge_texts]

    return knowledge_texts


@KnowledgeExtractionNameSpace.register("FOLIO_NL")
def extract_knowledge_texts(rationale: str) -> Union[set[str], list[str]]:
    knowledge_pattern = re.compile(r"(we|We)\s+(have|retrieve)(\s+the)?(\s+logic)?\s+knowledge(\s+that)?"
                                   r"\s+\"(.+?)\"[.,;:?! ]")
    knowledge_texts = knowledge_pattern.findall(rationale)
    knowledge_texts = [k[-1].strip() for k in knowledge_texts]

    return knowledge_texts

