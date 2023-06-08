from concurrent import futures
import grpc
import tokenize_pb2
import tokenize_pb2_grpc
from datasets import Dataset
import pandas as pd
import pyarrow as pa
from transformers import BartTokenizer

def encode(input, tokenizer):
  # Tokenize the input and target text
  tokenized_text = tokenizer.batch_encode_plus(input['article'], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
  # Return the tokenized input
  return {'input_ids': tokenized_text['input_ids'], 'attention_mask': tokenized_text['attention_mask']}

class TokenizerRPC(tokenize_pb2_grpc.TokenizerRPCServicer):
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def Tokenize(self, request, context):
        print('Received Tokenize Request')
        table = pa.Table.from_pandas(pd.DataFrame({'article': [request.text]}))
        # Create a dataset with the Arrow table
        my_dataset = Dataset(table)
        tokenized = my_dataset.map(encode, batched=True, fn_kwargs={'tokenizer': self.tokenizer})

        response = tokenize_pb2.TokenizeResponse(tokens= tokenized['input_ids'][0], attention_mask=tokenized['attention_mask'][0])
        print('Sending Response...')
        return response
    
    def Decode(self, request, context):
        print('Received Decode Request')
        outputs = request.tokens
        text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        print('Sending Response...')
        return tokenize_pb2.DecodeResponse(text=text)


def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tokenize_pb2_grpc.add_TokenizerRPCServicer_to_server(TokenizerRPC(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
