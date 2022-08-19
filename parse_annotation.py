# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + code_folding=[38, 49]
### works for twitter data
def parse_annotation(input_file_path, annotated_file_path, extract_16=True, check_intertweet=False):
    extract = None
    if extract_16:
        extract = extract_from_ranges_16
    else:
        extract = extract_from_ranges
    annotations = []
    with open(input_file_path, encoding='utf8') as input_file:
        with open(annotated_file_path) as ann_file:
            try:
                file_content = input_file.read()#decode('ISO-8859-1')
            except Exception as inst:
                print(type(inst))
            for line in ann_file:
                properties = line.split('|')
                typ = properties[0]
                connective_ranges = None
                connective = None
                if typ == 'Explicit':
                    connective_ranges = properties[1]
                    try:
                        connective = extract(file_content, connective_ranges)
                    except Exception as inst:
                        connective = '--ERROR--'
                if typ == 'Implicit' or typ == 'Hypophora':
                    connective_ranges = properties[31]
                sense = properties[8]
                arg1_range = properties[14]
                arg2_range = properties[20]
                
                intertweet = None
                if check_intertweet:
                    try:
                        intertweet = is_intertweet([arg1_range, arg2_range, connective_ranges], file_content)
                    except Exception as inst:
                        interweet = '--ERROR--'
                try:
                    arg1 = extract(file_content, arg1_range)
                except Exception as inst:
                    arg1 = '--ERROR--'
                try:
                    arg2 = extract(file_content, arg2_range)
                except Exception as inst:
                    arg2 = '--ERROR--'
                result = {
                    'file_id': input_file_path,
                    'arg1_range': arg1_range,
                    'arg2_range': arg2_range,
                    'arg1': arg1,
                    'arg2': arg2,
                    'connective_range': connective_ranges,
                    'connective': connective,
                    'sense': sense,
                    'type': typ,
                    'intertweet': intertweet
                }
                if check_intertweet:
                    result['intertweet'] = intertweet
                annotations.append(result)
    return annotations

def parse_annotation_pure(annotated_file_path):
    annotations = []
    with open(annotated_file_path) as ann_file:
        for line in ann_file:
            properties = line.split('|')
            typ = properties[0]
            connective_ranges = None
            if typ == 'Explicit':
                connective_ranges = properties[1]
            if typ == 'Implicit' or typ == 'Hypophora':
                connective_ranges = properties[31]
            sense = properties[8]
            arg1_range = properties[14]
            arg2_range = properties[20]

            result = {
                'file_id': annotated_file_path,
                'arg1_range': arg1_range,
                'arg2_range': arg2_range,
                'connective_range': connective_ranges,
                'sense': sense,
                'type': typ
            }
            annotations.append(result)
    return annotations

    

def parse_tweets(file_id, annotations, raw_file_folder, input_file_folder):
    tweets = []
    with open(raw_file_folder + '/' + file_id, 'rb') as raw_file:
        with open(input_file_folder + '/' + file_id, encoding='utf8') as input_file:
            raw_file_text = raw_file.read()
            input_file_text = input_file.read()
            raw_utf8 = raw_file_text.decode('utf-8')
            raw_iso = input_file_text
            whole_text = raw_iso.encode('utf-16le')
            raw_file_lines = raw_utf8.split('\n')
            tweet_texts = []
            for raw_file_line in raw_file_lines[:-1]:
                tokens = raw_file_line.split('\t')[1].split()
                tokens_to_be_removed = []
                for token in tokens:
                    if token[0] == '@':
                        tokens_to_be_removed.append(token)
                    else:
                        break
                for token in tokens_to_be_removed:
                    tokens.remove(token)
                tweet_texts.append(' '.join(tokens))
            # :-1 because raw files has extra one empty line
            for line_number, line in enumerate(raw_file_lines[:-1]):
                poster = line.split('\t')[0]
                tweet_text = tweet_texts[line_number]
                tweets.append({
                    'poster': poster,
                    'tweet_text': tweet_text,
                    'arg1': set(),
                    'arg2': set(),
                    'connective': set(),
                })

            for annotation_index, annotation in enumerate(annotations):
                # here, the annotations are going to be identified with their indexes in the annotation file
                # arg1
                for i, arg1_range_part in enumerate(annotation['arg1_range'].split(';')):
                    if len(arg1_range_part) > 0:
                        arg1_range_part_start = int(arg1_range_part.split('..')[0])
                        arg1_range_part_end = int(arg1_range_part.split('..')[1])
                        start_line = whole_text[:arg1_range_part_start * 2].decode('utf-16le').count('\n')
                        end_line = whole_text[:arg1_range_part_end * 2].decode('utf-16le').count('\n')
                        tweets[start_line]['arg1'].add(annotation_index)
                        tweets[end_line]['arg1'].add(annotation_index)

                # arg2
                for i, arg2_range_part in enumerate(annotation['arg2_range'].split(';')):
                    if len(arg2_range_part) > 0:
                        arg2_range_part_start = int(arg2_range_part.split('..')[0])
                        arg2_range_part_end = int(arg2_range_part.split('..')[1])
                        start_line = whole_text[:arg2_range_part_start * 2].decode('utf-16le').count('\n')
                        end_line = whole_text[:arg2_range_part_end * 2].decode('utf-16le').count('\n')
                        tweets[start_line]['arg2'].add(annotation_index)
                        tweets[end_line]['arg2'].add(annotation_index)

                # connective
                if (annotation['type'] == 'Explicit'):
                    for i, connective_range_part in enumerate(annotation['connective_range'].split(';')):
                        connective_range_part_start = int(connective_range_part.split('..')[0])
                        start_line = whole_text[:connective_range_part_start * 2].decode('utf-16le').count('\n')
                        tweets[start_line]['connective'].add(annotation_index)

                        connective_range_part_end = int(connective_range_part.split('..')[1])
                        end_line = whole_text[:connective_range_part_end * 2].decode('utf-16le').count('\n')
                        tweets[end_line]['connective'].add(annotation_index)
    return tweets
    
def extract_from_ranges_16(file_content, rangs):
    file_content = file_content.encode('utf-16le')
    text = []
    for rang in rangs.split(';'):
        if len(rang) < 2:
            return ''
        splitted = rang.split('..')
        part_text = file_content[int(splitted[0]) * 2: int(splitted[1]) * 2].decode('utf-16le')
        text.append(part_text)
        
    return text

def extract_from_ranges(file_content, rangs):
    text = []
    for rang in rangs.split(';'):
        if len(rang) < 2:
            return ''
        splitted = rang.split('..')
        part_text = file_content[int(splitted[0]): int(splitted[1])]
        text.append(part_text)
        
    return text

def is_intertweet(ranges, file_content):
    offsets = []
    for _range in ranges:
        if _range == None:
            continue
        for cur_offsets in _range.split(';'):
            if len(cur_offsets) < 2:
                return ''
            splitted = cur_offsets.split('..')
            for offset in splitted:
                offsets.append(offset)
    min_offset = min(offsets)
    max_offset = max(offsets)
    
    file_content = file_content.encode('utf-16le')
    substr = file_content[int(min_offset) * 2: int(max_offset) * 2].decode('utf-16le')
    i = substr.find('\n')
    return i >= 0



# +
import json

### works for PDTB
def parse_relations(relations_file_path):
    relations = []
    with open(relations_file_path, encoding="utf-8") as relations_file:
        for relation_line in relations_file.readlines():
            relation_json = json.loads(relation_line)
            relation = {
                'arg1': relation_json['Arg1']['RawText'],
                'arg2': relation_json['Arg2']['RawText'],
                'connective': relation_json['Connective']['RawText'],
                'sense': relation_json['Sense'][0],
                'type': relation_json['Type']
            }
            relations.append(relation)
    return relations
