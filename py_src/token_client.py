import grpc
import tokenize_pb2
import tokenize_pb2_grpc


# input_text = 'CNN)Governments around the world are using the threat of terrorism -- real or perceived -- to advance executions, Amnesty International alleges in its annual report on the death penalty. "The dark trend of governments using the death penalty in a futile attempt to tackle real or imaginary threats to state security and public safety was stark last year," said Salil Shetty, Amnesty\'s Secretary General in a release. "It is shameful that so many states around the world are essentially playing with people\'s lives -- putting people to death for \'terrorism\' or to quell internal instability on the ill-conceived premise of deterrence." The report, "Death Sentences and Executions 2014," cites the example of Pakistan lifting a six-year moratorium on the execution of civilians following the horrific attack on a school in Peshawar in December. China is also mentioned, as having used the death penalty as a tool in its "Strike Hard" campaign against terrorism in the restive far-western province of Xinjiang.'
input_text = 'This is a test'

with grpc.insecure_channel('localhost:50051') as channel:
    stub = tokenize_pb2_grpc.TokenizerRPCStub(channel)
    request = tokenize_pb2.TokenizeRequest()
    request.text = input_text
    response = stub.Tokenize(request)

input_ids = response.tokens 
attention_mask = response.attention_mask
print(input_ids)

with grpc.insecure_channel('localhost:50051') as channel:
    stub = tokenize_pb2_grpc.TokenizerRPCStub(channel)
    req = tokenize_pb2.DecodeRequest(tokens=[10,20,50,60,70,50])
    response = stub.Decode(req)
print("Tokenizer client received: " + response.text)
