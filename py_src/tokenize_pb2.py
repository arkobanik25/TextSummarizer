# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tokenize.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0etokenize.proto\"\x1f\n\x0fTokenizeRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\":\n\x10TokenizeResponse\x12\x0e\n\x06tokens\x18\x01 \x03(\x05\x12\x16\n\x0e\x61ttention_mask\x18\x02 \x03(\x05\"\x1f\n\rDecodeRequest\x12\x0e\n\x06tokens\x18\x01 \x03(\x05\"\x1e\n\x0e\x44\x65\x63odeResponse\x12\x0c\n\x04text\x18\x01 \x01(\t2j\n\x0cTokenizerRPC\x12/\n\x08Tokenize\x12\x10.TokenizeRequest\x1a\x11.TokenizeResponse\x12)\n\x06\x44\x65\x63ode\x12\x0e.DecodeRequest\x1a\x0f.DecodeResponseb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tokenize_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TOKENIZEREQUEST._serialized_start=18
  _TOKENIZEREQUEST._serialized_end=49
  _TOKENIZERESPONSE._serialized_start=51
  _TOKENIZERESPONSE._serialized_end=109
  _DECODEREQUEST._serialized_start=111
  _DECODEREQUEST._serialized_end=142
  _DECODERESPONSE._serialized_start=144
  _DECODERESPONSE._serialized_end=174
  _TOKENIZERRPC._serialized_start=176
  _TOKENIZERRPC._serialized_end=282
# @@protoc_insertion_point(module_scope)
