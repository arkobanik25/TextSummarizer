from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DecodeRequest(_message.Message):
    __slots__ = ["tokens"]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class DecodeResponse(_message.Message):
    __slots__ = ["text"]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class TokenizeRequest(_message.Message):
    __slots__ = ["text"]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class TokenizeResponse(_message.Message):
    __slots__ = ["attention_mask", "tokens"]
    ATTENTION_MASK_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    attention_mask: _containers.RepeatedScalarFieldContainer[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, tokens: _Optional[_Iterable[int]] = ..., attention_mask: _Optional[_Iterable[int]] = ...) -> None: ...
