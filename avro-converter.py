
# Import the schema, datafile and io submodules
# from avro (easy_install avro)
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

def read_avro_file(name):
    # Create a 'record' (datum) reader
    # You can pass an 'expected=SCHEMA' kwarg
    # if you want it to expect a particular
    # schema (Strict)
    rec_reader = io.DatumReader()
 
    # Create a 'data file' (avro file) reader
    df_reader = datafile.DataFileReader(
                    open(name),
                    rec_reader
                )
 
    # Read all records stored inside
    for record in df_reader:
        with open(record['filename'], 'wb') as f:
           f.write(record['content'])

        #print record['filename'], record['age']
        #print record['address'], record['value']
        # Do whatever read-processing you wanna do
        # for each record here ...
 

# Create a 'record' (datum) writer
rec_writer = io.DatumWriter(SCHEMA)

df_writer = None

if os.path.isfile(sys.argv[1]):
  df_writer = datafile.DataFileWriter(
  		open(sys.argv[1], 'ab+'),
  		rec_writer,
  		codec = 'deflate'
  		)
else: 
  df_writer = datafile.DataFileWriter(
  		open(sys.argv[1], 'ab+'),
  		rec_writer,
  		writers_schema = SCHEMA,
  		codec = 'deflate'
  		)

write_avro_file(sys.argv[2], df_writer)

## Close to ensure writing is complete
df_writer.close()


