from typing import List, Optional, Union, Tuple

class LLMEngine:
    def generate(
        self,
        prompt: Optional[Union[str, List[int]]],
        max_tokens: int = 10240,
        temperature: float = 0.6,
        stream: bool = False,
    ) -> Tuple[List[int], str]:
        raise NotImplementedError
    
    def tokenize(self, prompt: str) -> List[int]:
        raise NotImplementedError
