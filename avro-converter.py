import sys
import os.path
from avro import schema, datafile, io

SCHEMA_STR = """{
    "type": "record",
    "name": "sampleAvro",
    "namespace": "AVRO",
    "fields": [
        {   "name": "filename"   , "type": "string"   },
        {   "name": "content"    , "type": "bytes"      }
    ]
}"""
 
SCHEMA = schema.parse(SCHEMA_STR)
 
def write_avro_file(name, writer):
    # Lets generate our data
    data = {}
    data['filename'] = name
    with open(name, 'rb') as f:
      data['content'] = f.read()


    # Write our data
    # (You can call append multiple times
    # to write more than one record, of course)
    writer.append(data)

# Create a 'record' (datum) writer
rec_writer = io.DatumWriter(SCHEMA)

df_writer = None

if os.path.isfile(sys.argv[2] + ".avro"):
  sys.exit(0)

else: 
  df_writer = datafile.DataFileWriter(
  		open(sys.argv[1] + "/" + sys.argv[2] + ".avro", 'wb+'),
  		rec_writer,
  		writers_schema = SCHEMA,
  		codec = 'deflate'
  		)

write_avro_file(sys.argv[2], df_writer)

## Close to ensure writing is complete
df_writer.close()


