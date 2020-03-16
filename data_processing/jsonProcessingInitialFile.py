import json
import csv

filei=open('iclr-results.csv', 'r')
fileo=open('processed_iclr.txt','w')
filei=csv.reader(filei,delimiter='\t')
filei=list(filei)
# fileo = csv.writer(fileo)
for handle in filei:
    # print(handle)
    parsed = json.loads(handle[0])
    fileo.write(json.dumps(parsed, indent=4))
    parsed = json.loads(handle[1])
    fileo.write(json.dumps(parsed, indent=4))
    fileo.write('\n')



# for record in reader:
#     # source_data
#     src = json.loads(record[0])
#     doc_id = src['uuid']
#     if doc_id not in dict_uuid_neg.keys():
#         dict_uuid_neg[doc_id] = []
#     topic_content = src['srcText']
#     topic, docid, content = topic_content.split('\n')

#     remove_digits = str.maketrans('', '', digits)
#     topic = topic.translate(remove_digits)

#     print(topic)
#     print(content)

#     # target_data
#     if not record[1]:
#         break
#     tgt = json.loads(record[1])
#     if not tgt:
#         break

#     if 'question^category' in tgt.keys():
#         stance = tgt['question^category']
#     elif 'classification' in tgt.keys():
#         stance = tgt['classification'][0]['value']

#     if stance.startswith('0'):
#         csv_writer.writerow([topic, topic, content, 0])
#         num_non += 1
#     if stance.startswith('1'):
#         csv_writer.writerow([topic, topic, content, 1])
#         num_positive += 1
#     if stance.startswith('2'):
#         csv_writer.writerow([topic, topic, content, 2])
#         num_negative += 1