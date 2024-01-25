from model import load_model, citation_correction
import argparse
from arguments import add_model_config_args
import os
import pandas as pd
import json

if __name__ == '__main__':
    #os.environ["SERPAPI_KEY"] = "SET YOUR KEY"
    os.environ["WEBGLM_RETRIEVER_CKPT"] = "./download/retriever-pretrained-checkpoint"
    arg = argparse.ArgumentParser()
    add_model_config_args(arg)
    args = arg.parse_args()
    
    webglm = load_model(args)
    ans_list=[]
    #file1 = open("myfile.txt", "a")
    with open('naturalQuestion.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        #first_5 = data[:2]
        #data = first_5
    for item in data:
        question = item['question']
        question = question.strip()
        if not question:
            break
        if question == "quit":
            break
        final_results = {}
        print(f"\n\nChecking for  question {question}\n")
        try:
            for results in webglm.stream_query(question):
                final_results.update(results)
                if "answer" in results:
                    print(results["answer"])
                    ans_1="%s"%citation_correction(results["answer"], [ref['text'] for ref in final_results["references"]])
                    ans_list.append([question,ans_1])
                    dictionary = {"query": question,"answer":ans_1}
                    json_object = json.dumps(dictionary, indent=4)
                    with open("WebGLMNaturalQuestionsAnswer.json", "a") as outfile:
                        outfile.write(json_object)
                        outfile.write(",\n")
        except:
            ans_1="Error"
            print(f"error in question {question}\n")
            dictionary = {"query": question,"answer":ans_1}
            json_object = json.dumps(dictionary, indent=4)
            with open("error.json", "a") as outfile:
                outfile.write(json_object)
                outfile.write(",\n")

    print(ans_list)
    a=pd.DataFrame(ans_list,columns=['question','ans'])
    a.to_csv('out.csv')
    #file1.close()