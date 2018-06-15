import sys
from avro import schema, datafile, io

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

read_avro_file(sys.argv[1])
