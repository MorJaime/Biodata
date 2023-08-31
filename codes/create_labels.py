import os
import xml.etree.ElementTree as ET
import csv
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')

OMIZU_PATH = os.environ["OMIZU_PATH"]
UMINEKO_PATH = os.environ["UMINEKO_PATH"]
LABELS_PATH = os.environ["LABELS_PATH"]

def make_labels(paths, label_wr_dir = '/home/bob/storage/database/labels', fn_end = 17):
    
    for path in paths:
        label_path = path
        label_dir_p, label_fn = os.path.split(label_path)
        wr_fn = label_fn[:fn_end]
        tree = ET.parse(label_path)
        root = tree.getroot()
        filename = os.path.join(label_wr_dir,wr_fn+'_labels.csv')
        
        with open(filename,"w") as f:            
            csv_writer = csv.writer(f)
            header = ["event_type","start", "end"]
            csv_writer.writerow(header)
            for labellist in root.iter("labellist"):
                timestampStart = labellist[1].text
                timestampStart = timestampStart.replace('-','')
                timestampStart = timestampStart.replace('T',' ')      
                timestampStart = timestampStart.replace('Z','')
                timestampEnd = labellist[2].text
                timestampEnd = timestampEnd.replace('-','')
                timestampEnd = timestampEnd.replace('T',' ')
                timestampEnd = timestampEnd.replace('Z','')

                row = [labellist[0].text, labellist[1].text, labellist[2].text]
                row = [labellist[0].text,timestampStart,timestampEnd]
                csv_writer.writerow(row)
            
        print('created labels for >>> ',filename)
        
    return

def find_xml_filenames(path_to_dir, suffix=".xml"):
    filenames = os.listdir(path_to_dir)
    filepaths = []
    for filename in filenames:
        if filename.endswith( suffix ):
            filepaths.append(os.path.join(path_to_dir,filename))
    return filepaths

def main(args):
    outdir = str(args['out_dir'])
    label_path_o = os.path.join(OMIZU_PATH,'labels')
    label_path_u = os.path.join(UMINEKO_PATH,'labels')
    
    if outdir == "None":
        wrdir = LABELS_PATH
        print(wrdir)
    else:
        wrdir = outdir
    
    label_paths_o = find_xml_filenames(label_path_o)
    label_paths_u = find_xml_filenames(label_path_u)

    make_labels(label_paths_o, label_wr_dir = wrdir, fn_end = 17)
    make_labels(label_paths_u, label_wr_dir = wrdir, fn_end = 11)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', help="Path to the output directory")

    args = vars(parser.parse_args())
    main(args)