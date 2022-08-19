
import json
import logging
import os
from collections import defaultdict

import click

from discopy.parsers.pipeline import ParserPipeline
from discopy.utils import init_logger
# from discopy_data.data.loaders.conll import load_bert_conll_dataset
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation
from discopy_data.data.sentence import DepRel, Sentence
from discopy_data.data.token import Token
from discopy_data.data.loaders.conll import load_bert_embeddings, convert_sense, connective_head
from discopy_data.data.relation import Relation

def load_bert_conll_dataset(conll_path: str, cache_dir='', bert_model='bert-base-cased',
                            simple_connectives=True, sense_level=-1):
    parses_path = os.path.join(conll_path, 'parses.json')
    relations_path = os.path.join(conll_path, 'relations.json')

    pdtb = defaultdict(list)

    all_docs = json.load(open(parses_path))
    docs = []
    for doc_id, doc in all_docs.items():
        doc['docID'] = doc_id
        if doc_id[:3] == 'wsj':
            if 0 < 0 < len(docs):
                break
            words = []
            token_offset = 0
            sents = []
            for sent_i, sent in enumerate(doc['sentences']):
                sent_words = [
                    Token(token_offset + w_i, sent_i, w_i, t['CharacterOffsetBegin'], t['CharacterOffsetEnd'], surface,
                        xpos=t['PartOfSpeech'])
                    for w_i, (surface, t) in enumerate(sent['words'])
                ]
                words.extend(sent_words)
                token_offset += len(sent_words)
                dependencies = [
                    DepRel(rel=rel,
                        head=sent_words[int(head.split('-')[-1]) - 1] if not head.startswith('ROOT') else None,
                        dep=sent_words[int(dep.split('-')[-1]) - 1]
                        ) for rel, head, dep in sent['dependencies']
                ]
                sents.append(Sentence(sent_words, dependencies=dependencies, parsetree=sent['parsetree']))
            doc_pdtb = pdtb.get(doc_id, [])
            if simple_connectives:
                connective_head(doc_pdtb)
            relations = [
                Relation([words[i[2]] for i in rel['Arg1']['TokenList']],
                        [words[i[2]] for i in rel['Arg2']['TokenList']],
                        [words[i[2]] for i in rel['Connective']['TokenList']],
                        [convert_sense(s, sense_level) for s in rel['Sense']], rel['Type']) for rel in doc_pdtb
            ]
            docs.append(Document(doc_id=doc_id, sentences=sents, relations=relations))
        else:
            docs.append(Document.from_json(doc))

    for relation in [json.loads(s) for s in open(relations_path, 'r').readlines()]:
        relation['Arg1']['RawText'] = ''.join(relation['Arg1']['RawText'])
        relation['Arg2']['RawText'] = ''.join(relation['Arg2']['RawText'])
        relation['Connective']['RawText'] = ''.join(relation['Connective']['RawText'])
        pdtb[relation['DocID']].append(relation)

    for doc in docs:
        words = doc.get_tokens()
        doc_pdtb = pdtb.get(doc.doc_id, [])
        if simple_connectives:
            connective_head(doc_pdtb)
        relations = [
            Relation([words[i[2]] for i in rel['Arg1']['TokenList']],
                     [words[i[2]] for i in rel['Arg2']['TokenList']],
                     [words[i[2]] for i in rel['Connective']['TokenList']],
                     [convert_sense(s, sense_level) for s in rel['Sense']], rel['Type']) for rel in doc_pdtb
        ]
        doc.relations = relations

    logging.info(f'Load {bert_model} Embeddings')
    docs = load_bert_embeddings(docs, cache_dir, bert_model)
    return docs

logger = logging.getLogger('discopy')


@click.command()
@click.argument('bert-model', type=str)
@click.argument('model-path', type=str)
@click.argument('conll-path', type=str)
@click.option('--cache-path', default='', type=str)
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(bert_model, conll_path, cache_path, model_path, tgt):
    logger = init_logger()
    logger.info('Load data')
    dataset_part = os.path.basename(conll_path)
    docs = load_bert_conll_dataset(conll_path,
                                   cache_dir=os.path.join(cache_path, f'{dataset_part}.{bert_model}.joblib'),
                                   bert_model=bert_model)
    logger.info('Init Parser')
    parser = ParserPipeline.from_config(model_path)
    logger.info('LOAD model')
    parser.load(model_path)
    logger.info(f'Predict on dataset: {conll_path}')
    for doc in docs:
        doc = parser.parse_doc(doc)
        tgt.write(json.dumps(doc.to_json()) + '\n')


if __name__ == '__main__':
    main()

