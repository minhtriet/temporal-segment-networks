import os
import pdb
import string
import sys
import subprocess

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Usage: path_flow replace_this_class with_this_class"
        sys.exit(0)
   
    sub_path = sys.argv[1]
    count = 0
    for sub_r, sub_s, sub_files in os.walk(sub_path):
        for aVid in sub_s:
            class_name = aVid.rsplit('_', 1)[1]
            if class_name == sys.argv[2]:
                old_name = os.path.join(sys.argv[1], aVid)
                new_name = os.path.join(sys.argv[1],"_".join([aVid.rsplit('_', 1)[0], sys.argv[3]]))
                subprocess.call("mv %s %s " % (old_name, new_name), shell = True)
                count = count + 1
                print("%s -> %s" % (old_name, new_name))
    print "Moved %d folder" % count


